extern crate typed_arena;

use std::collections::btree_map::BTreeMap;
use std::collections::btree_set::BTreeSet;
use std::cell::{Ref, RefMut, RefCell};
use typed_arena::Arena;

#[derive(Debug, Default)]
struct RefCellWrapper<T> {
    pub refcell: RefCell<T>,
}

impl<T> RefCellWrapper<T> {
    fn new(inner: T) -> RefCellWrapper<T> {
        RefCellWrapper {
            refcell: RefCell::new(inner),
        }
    }

    fn borrow(&self) -> Ref<T> {
        self.refcell.borrow()
    }

    fn borrow_mut(&self) -> RefMut<T> {
        self.refcell.borrow_mut()
    }
}

impl<T> PartialEq<RefCellWrapper<T>> for RefCellWrapper<T> {
    fn eq(&self, other: &RefCellWrapper<T>) -> bool {
        // reference equality instead of value equality
        self as *const _ == other as *const _
    }
}

impl<T> Eq for RefCellWrapper<T> {}

impl<T> PartialOrd<RefCellWrapper<T>> for RefCellWrapper<T> {
    fn partial_cmp(&self, other: &RefCellWrapper<T>) -> Option<std::cmp::Ordering> {
        // comparison based on memory address
        Some((self as *const _).cmp(&(other as *const _)))
    }
}

impl<T> Ord for RefCellWrapper<T> {
    fn cmp(&self, other: &RefCellWrapper<T>) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

type LoopRef<'a, 'b> = RefCellWrapper<SimpleLoop<'a, 'b>>;
type BBRef<'b> = RefCellWrapper<BasicBlock<'b>>;
type UFRef<'a, 'b, 'c> = RefCellWrapper<UnionFindNode<'a, 'b, 'c>>;

struct BasicBlock<'a> {
    pub name: i32,
    pub in_edges: Vec<&'a BBRef<'a>>,
    pub out_edges: Vec<&'a BBRef<'a>>,
}

impl<'a> BasicBlock<'a> {
    fn new(name: i32) -> BasicBlock<'a> {
        BasicBlock {
            name: name,
            in_edges: Vec::new(),
            out_edges: Vec::new(),
        }
    }
}

struct BasicBlockEdge<'a> {
    to: &'a BBRef<'a>,
    from: &'a BBRef<'a>,
}

impl<'a> BasicBlockEdge<'a> {
    fn new(cfg: &mut CFG<'a>, from: i32, to: i32) {
        let bbe = BasicBlockEdge {
            to: cfg.create_node(to),
            from: cfg.create_node(from),
        };

        bbe.from.borrow_mut().out_edges.push(bbe.to);
        bbe.to.borrow_mut().in_edges.push(bbe.from);

        cfg.add_edge(bbe);
    }
}

struct CFG<'a> {
    bb: BTreeMap<i32, &'a BBRef<'a>>,
    edge_list: Vec<BasicBlockEdge<'a>>,
    start_node: Option<&'a BBRef<'a>>,
    arena: &'a Arena<BBRef<'a>>
}

impl<'a> CFG<'a> {
    fn new(arena: &'a Arena<BBRef<'a>>) -> CFG<'a> {
        CFG {
            bb: Default::default(),
            edge_list: Vec::new(),
            start_node: None,
            arena: arena,
        }
    }

    fn new_edge(&mut self, from: i32, to: i32) {
        BasicBlockEdge::new(self, from, to);
    }

    fn create_node(&mut self, node: i32) -> &'a BBRef<'a> {
        let bblock: &BBRef = self.bb.entry(node)
                                 .or_insert(self.arena.alloc(RefCellWrapper::new(BasicBlock::new(node))));

        if self.bb.len() == 1 {
            self.start_node = Some(bblock);
        }
        bblock
    }

    fn add_edge(&mut self, edge: BasicBlockEdge<'a>) {
        self.edge_list.push(edge);
    }

    fn get_num_nodes(&self) -> usize {
        self.bb.len()
    }

    fn get_start_basic_block(&self) -> Option<&'a BBRef<'a>> {
        self.start_node
    }

    fn get_basic_blocks(&self) -> &BTreeMap<i32, &'a BBRef<'a>> {
        &self.bb
    }
}

struct SimpleLoop<'a, 'b: 'a> {
    basic_blocks: BTreeSet<&'b BBRef<'b>>,
    children: BTreeSet<&'a LoopRef<'a, 'b>>,
    parent: Option<&'a LoopRef<'a, 'b>>,
    is_root: bool,
    is_reducible: bool,
    counter: i32,
    nesting_level: i32,
    depth_level: i32
}

impl<'a, 'b> SimpleLoop<'a, 'b> {
    fn new() -> SimpleLoop<'a, 'b> {
        SimpleLoop {
            basic_blocks: Default::default(),
            children: Default::default(),
            parent: None,
            is_root: false,
            is_reducible: true,
            counter: 0,
            nesting_level: 0,
            depth_level: 0,
        }
    }

    fn add_node(&mut self, bb: &'b BBRef<'b>) {
        self.basic_blocks.insert(bb);
    }

    fn add_child_loop(&mut self, ch_loop: &'a LoopRef<'a, 'b>) {
        self.children.insert(ch_loop);
    }

    fn set_parent(this: &'a LoopRef<'a, 'b>, parent: &'a LoopRef<'a, 'b>) {
        this.borrow_mut().parent = Some(parent);
        parent.borrow_mut().add_child_loop(this);
    }

    fn set_nesting_level(&mut self, level: i32) {
        self.nesting_level = level;
    }

    fn set_counter(&mut self, c: i32) {
        self.counter = c;
    }
}

struct LSG<'a, 'b: 'a> {
    loop_arena: &'a Arena<LoopRef<'a, 'b>>,
    loops: Vec<&'a LoopRef<'a, 'b>>,
    root: Option<&'a LoopRef<'a, 'b>>,
    loop_counter: i32,
}

impl<'a, 'b> LSG<'a, 'b> {
    fn new(loop_arena: &'a Arena<LoopRef<'a, 'b>>) -> LSG<'a, 'b> {
        let mut lsg = LSG {
            loop_arena: loop_arena,
            loops: vec![],
            root: None,
            loop_counter: 0,
        };

        let root = lsg.create_new_loop();
        root.borrow_mut().set_nesting_level(0);
        lsg.root = Some(root);

        lsg
    }

    fn create_new_loop(&mut self) -> &'a LoopRef<'a, 'b> {
        let mut s = SimpleLoop::new();
        s.set_counter(self.loop_counter);
        self.loop_counter += 1;
        let r = self.loop_arena.alloc(RefCellWrapper::new(s));
        self.add_loop(r);
        r
    }

    fn add_loop(&mut self, l :&'a LoopRef<'a, 'b>) {
        self.loops.push(l);
    }
}

struct UnionFindNode<'a, 'b: 'a, 'c: 'a + 'b> {
    parent: Option<&'a UFRef<'a, 'b, 'c>>,
    bb: Option<&'c BBRef<'c>>,
    s_loop: Option<&'b LoopRef<'b, 'c>>,
    dfs_number: i32,
}

impl<'a, 'b, 'c> UnionFindNode<'a, 'b, 'c> {
    fn new() -> UnionFindNode<'a, 'b, 'c> {
        UnionFindNode {
            parent: None,
            bb: None,
            s_loop: None,
            dfs_number: 0,
        }
    }

    fn init(&mut self, bb: &'c BBRef<'c>, dfs: i32) {
        self.parent = None;
        self.bb = Some(bb);
        self.s_loop = None;
        self.dfs_number = dfs;
    }

    fn find_set(mut this: &'a UFRef<'a, 'b, 'c>) -> &'a UFRef<'a, 'b, 'c> {
        let mut nodelist = vec![];

        while let Some(x) = {
            let cl = this.borrow().parent;
            if Some(this) == cl { None } else { cl }
        } {
            if let Some(y) = x.borrow().parent {
                if x != y {
                    nodelist.push(this);
                }
            }
            this = x;
        }

        for e in &nodelist {
            let p = this.borrow().parent;
            e.borrow_mut().set_parent(p);
        }
        this
    }

    fn union(&mut self, b: &'a UFRef<'a, 'b, 'c>) {
        self.set_parent(Some(b));
    }

    fn set_parent(&mut self, parent: Option<&'a UFRef<'a, 'b, 'c>>) {
        self.parent = parent;
    }

    fn get_bb(&self) -> Option<&'c BBRef<'c>> {
        self.bb
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BasicBlockClass {
    Top,
    Nonheader,
    Reducible,
    IsSelf, // Self is a keyword in Rust
    Irreducible,
    Dead,
    Last,
}

const K_UNVISITED: i32 = std::i32::MAX;
const K_MAX_NON_BACK_PREDS: i32 = 32 * 1024;

struct HavlakLoopFinder<'a: 'c, 'b: 'a, 'c> {
    cfg: &'a mut CFG<'b>,
    lsg: LSG<'c, 'b>
}

impl<'a, 'b, 'c> HavlakLoopFinder<'a, 'b, 'c>  {
    fn new(cfg: &'a mut CFG<'b>, lsg: LSG<'c, 'b>) -> HavlakLoopFinder<'a, 'b, 'c> {
        HavlakLoopFinder {
            cfg: cfg,
            lsg: lsg
        }
    }

    fn is_ancestor(w: i32, v: i32, arr: &[i32]) -> bool {
        w <= v && v <= arr[w as usize]
    }

    fn dfs<'x, 'y, 'z>(current_node: &'z BBRef<'z>,
           nodes: &mut [&'x UFRef<'x, 'y, 'z>],
           number: &mut BTreeMap<&'z BBRef<'z>, i32>,
           last: &mut [i32],
           current: i32) -> i32 {

        nodes[current as usize].borrow_mut().init(current_node, current);
        number.insert(current_node, current);

        let mut last_id = current;
        let curr = current_node;
        for target in &curr.borrow().out_edges {
            if *number.get(target).unwrap() == K_UNVISITED {
                last_id = Self::dfs(target, nodes, number, last, last_id + 1);
            }
        }
        last[number[&current_node] as usize] = last_id;
        last_id
    }

    fn find_loops(mut self) -> i32 {
        if self.cfg.get_start_basic_block().is_none() { return 0; }
        let uf_arena = Arena::<UFRef>::new();

        let size = self.cfg.get_num_nodes();
        let non_back_preds: Vec<RefCell<BTreeSet<i32>>> = vec![Default::default(); size];
        let mut back_preds: Vec<Vec<i32>> = vec![Vec::new(); size];
        let mut header = vec![0i32; size];
        let mut btype = vec![BasicBlockClass::Nonheader; size];
        let mut last = vec![0i32; size];
        let mut nodes: Vec<&UFRef> = Vec::with_capacity(size);
        let mut number: BTreeMap<&BBRef, i32> = Default::default();

        for _ in 0..size {
            nodes.push(uf_arena.alloc(RefCellWrapper::new(UnionFindNode::new())));
        }

        for b in self.cfg.get_basic_blocks() {
            number.insert(b.1, K_UNVISITED);
        }

        Self::dfs(self.cfg.get_start_basic_block().unwrap(), &mut nodes, &mut number, &mut last, 0);

        for w in 0..size {
            header[w] = 0;
            let nb = nodes[w].borrow();
            let node_w = nb.get_bb();
            if let Some(x) = node_w {
                for node_v in &x.borrow().in_edges {
                    let v = number[node_v];
                    if v == K_UNVISITED { continue; }

                    if Self::is_ancestor(w as i32, v, &last) {
                        back_preds[w].push(v);
                    } else {
                        non_back_preds[w].borrow_mut().insert(v);
                    }
                }
            } else {
                btype[w] = BasicBlockClass::Dead;
            }
        }

        header[0] = 0;

        for w in (0..size).rev() {
            let mut node_pool: Vec<&UFRef> = vec![];

            if nodes[w].borrow_mut().get_bb().is_none() {
                continue;
            }

            for v in &back_preds[w] {
                if *v == w as i32 {
                    btype[w] = BasicBlockClass::IsSelf;
                } else {
                    node_pool.push(UnionFindNode::find_set(nodes[*v as usize]));
                }
            }

            let mut worklist: Vec<&UFRef> = node_pool.iter().rev().cloned().collect();

            if !node_pool.is_empty() {
                btype[w] = BasicBlockClass::Reducible;
            }

            while let Some(x) = worklist.pop() {
                let x_dfs = { x.borrow().dfs_number };
                let non_back_size = non_back_preds[x_dfs as usize].borrow().len();
                if non_back_size as i32 > K_MAX_NON_BACK_PREDS {
                    return 0;
                }

                for y in non_back_preds[x_dfs as usize].borrow().iter() {
                    let ydash = UnionFindNode::find_set(nodes[*y as usize]);

                    let num = ydash.borrow().dfs_number;
                    if Self::is_ancestor(w as i32, num, &last) {
                        if num != w as i32 && !node_pool.contains(&ydash) {
                            worklist.push(ydash);
                            node_pool.push(ydash);
                        }
                    } else {
                        btype[w] = BasicBlockClass::Irreducible;
                        non_back_preds[w].borrow_mut().insert(num);
                    }
                }
            }
            if !node_pool.is_empty() || btype[w] == BasicBlockClass::IsSelf {
                let mut nb2 = nodes[w].borrow_mut();
                let new_loop = self.lsg.create_new_loop();

                nb2.s_loop = Some(new_loop);

                for mut node in node_pool.iter().map(|x| x.borrow_mut()) {
                    header[node.dfs_number as usize] = w as i32;
                    node.union(nodes[w]);

                    if let Some(l) = node.s_loop {
                        if l != new_loop {
                            SimpleLoop::set_parent(l, new_loop);
                        }
                    } else {
                        let mut br = new_loop.borrow_mut();
                        br.add_node(node.bb.unwrap());
                    }
                }
            }
        }
        self.lsg.loop_counter
    }
}

fn find_havlak_loops<'a>(cfg: &mut CFG<'a>) -> i32 {
    let a = Arena::new();
    let l = LSG::new(&a);

    let finder = HavlakLoopFinder::new(cfg, l);
    finder.find_loops()
}

fn build_diamond<'a>(cfg: &mut CFG<'a>, start: i32) -> i32 {
    let bb0 = start;

    cfg.new_edge(bb0, bb0 + 1);
    cfg.new_edge(bb0, bb0 + 2);
    cfg.new_edge(bb0 + 1, bb0 + 3);
    cfg.new_edge(bb0 + 2, bb0 + 3);

    bb0 + 3
}

fn build_connect<'a>(cfg: &mut CFG<'a>, start: i32, end: i32) {
    cfg.new_edge(start, end);
}

fn build_straight<'a>(cfg: &mut CFG<'a>, start: i32, n: i32) -> i32 {
    for i in 0..n {
        build_connect(cfg, start + i, start + i + 1);
    }

    start + n
}

fn build_base_loop<'a>(cfg: &mut CFG<'a>, from: i32) -> i32 {
    let header = build_straight(cfg, from, 1);
    let diamond1 = build_diamond(cfg, header);
    let d11 = build_straight(cfg, diamond1, 1);
    let diamond2 = build_diamond(cfg, d11);
    let mut footer = build_straight(cfg, diamond2, 1);
    build_connect(cfg, diamond2, d11);
    build_connect(cfg, diamond1, header);

    build_connect(cfg, footer, from);
    footer = build_straight(cfg, footer, 1);
    footer
}

fn main() {
    use std::io::Write;
    let arena = Arena::new();
    let mut cfg = CFG::new(&arena);
    {
        cfg.create_node(0);
        build_base_loop(&mut cfg, 0);
        cfg.create_node(1);
        cfg.new_edge(0, 2);
    }

    for _ in 0..15000usize {
        find_havlak_loops(&mut cfg);
    }

    let mut n = 2i32;

    for _ in 0..10usize {
        cfg.create_node(n + 1);
        build_connect(&mut cfg, 2, n + 1);
        n += 1;

        for _ in 0..100usize {
            let top = n;
            n = build_straight(&mut cfg, n, 1);
            for _ in 0..25usize {
                n = build_base_loop(&mut cfg, n);
            }

            let bottom = build_straight(&mut cfg, n, 1);
            build_connect(&mut cfg, n, top);
            n = bottom;
        }
        build_connect(&mut cfg, n, 1);
    }

    let num = find_havlak_loops(&mut cfg);

    let mut sum = 0i32;
    for _ in 0..50usize {
        write!(&mut std::io::stderr(), ".").unwrap();
        sum += find_havlak_loops(&mut cfg);
    }
    writeln!(&mut std::io::stderr(), "").unwrap();
    writeln!(&mut std::io::stderr(), "{:?}, {:?}", num, sum).unwrap();
}
