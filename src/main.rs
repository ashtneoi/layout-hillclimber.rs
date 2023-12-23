use lazy_static::lazy_static;
use num_format::WriteFormatted;
use rand::prelude::*;
use rand::distributions::Uniform;
use signal_hook::consts::signal::{SIGINT, SIGTERM};
use signal_hook::flag;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, prelude::*, BufReader, ErrorKind};
use std::process::exit;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

type Layout = Vec<Vec<u8>>;
type Ngrams = Vec<Vec<(String, u64)>>;

lazy_static! {
    static ref PLEASE_STOP: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
}

// -2 = painful or very slow
// -1 = unpleasant or slow
// 0 = neutral
// 1 = fast and comfortable
// 2 = very fast and comfortable
static MOVEMENT_TABLE: &'static [[[i64; 5]; 5]; 5] = &[
    [ // up 2
        [-2, -2, 0, -1, -1], // pinky -> ?
        [-2, -2, -1, -2, -2], // ring -> ?
        [-2, -2, -2, -1, -2], // middle -> ?
        [-1, -2, -1, -2, -2], // index -> ?
        [-2, -2, -2, -2, -2], // stretched index -> ?
    ],
    [ // up 1
        [-2, 0, 0, 0, -1], // pinky -> ?
        [-1, -1, 1, 0, -2], // ring -> ?
        [-1, -1, -1, -1, -2], // middle -> ?
        [0, 0, 1, -1, -1], // index -> ?
        [-1, -2, -2, -2, -2], // stretched index -> ?
    ],
    [ // level
        [-1, 2, 2, 2, 0], // pinky -> ?
        [1, 0, 2, 2, -2], // ring -> ?
        [1, 1, 0, 2, -2], // middle -> ?
        [1, 1, 2, 0, -1], // index -> ?
        [0, -2, -2, -1, 0], // stretched index -> ?
    ],
    [ // down 1
        [-2, -2, -1, 0, -2], // pinky -> ?
        [-1, -2, -1, 2, -2], // ring -> ?
        [1, 0, -1, 2, -2], // middle -> ?
        [0, -1, -1, -1, -2], // index -> ?
        [-1, -2, -2, -2, -2], // stretched index -> ?
    ],
    [ // down 2
        [-2, -2, -2, -1, -2], // pinky -> ?
        [-2, -2, -2, -2, -2], // ring -> ?
        [0, -2, -2, -1, -2], // middle -> ?
        [-1, -2, -2, -2, -2], // index -> ?
        [-2, -2, -2, -2, -2], // stretched index -> ?
    ],
];

fn get_ngrams(maxlen: usize, whole_only: bool) -> Ngrams {
    let mut n = vec![vec![]];
    let f = File::open("ngrams-all.tsv").unwrap();
    let mut lines = BufReader::new(f);
    let mut line = String::new();
    for i in 1..=maxlen {
        if line.is_empty() {
            lines.read_line(&mut line).unwrap();
        }
        assert!(!line.is_empty());
        let fields: Vec<_> = line.splitn(4, '\t').collect();
        let kind = fields[0];
        assert_eq!(kind, &format!("{}-gram", i));
        let splats;
        if whole_only {
            splats = fields[2];
            assert_eq!(splats, &format!("{}/*", i));
        } else {
            splats = fields[1];
            assert_eq!(splats, "*/*");
        }
        line.clear();
        let mut igrams = Vec::new();
        while lines.read_line(&mut line).unwrap() > 0 {
            let fields: Vec<_> = line.splitn(4, '\t').collect();
            let igram = fields[0];
            if igram.ends_with("-gram") {
                // Don't clear line (it's the next header).
                break;
            }
            let count = fields[if whole_only { 2 } else { 1 }];
            igrams.push((igram.to_string(), count.parse().unwrap()));
            line.clear();
        }
        n.push(igrams);
    }

    n
}

static COL_COUNT: usize = 10;
static COL_HALF: usize = COL_COUNT / 2;

static KEY_TO_STRENGTH: &[&[i64]] = &[
    &[0, 14, 18, 16, 8, 4, 16, 18, 14, 0],
    &[4, 16, 20, 18, 10, 10, 18, 20, 16, 4],
    &[0, 2, 2, 8, 4, 10, 16, 2, 2, 0],
];

fn strength_score(
    ngram: &str,
    count: u64,
    char_to_key: &HashMap<char, (usize, usize)>,
) -> i64 {
    let mut score = 0;
    for chr in ngram.chars() {
        let chr = chr as char;
        let (r, c) = char_to_key[&chr];
        let strength = KEY_TO_STRENGTH[r][c];
        score += strength * (count as i64);
    }
    score
}

fn finger_score_delta(
    r: usize, prev_r: usize,
    lc: usize, prev_lc: usize,
) -> i64 {
    MOVEMENT_TABLE[(r as isize - prev_r as isize + 2) as usize][prev_lc as usize][lc as usize]
        - if (r as isize - prev_r as isize).abs() == 2 { 2 } else { 0 }
        + 2
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Hand { Left, Right }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LKey {
    row: usize,
    lcol: usize,
}

fn movement_score(
    ngram: &str,
    count: u64,
    char_to_key: &HashMap<char, (usize, usize)>,
) -> (i64, i64) {
    let count = count as i64;
    let mut finger_score: i64 = 0;
    let mut cshu_score: i64 = 0;

    let mut other_prev_lk: Option<LKey> = None;
    let mut prev_lk: Option<LKey> = None;
    let mut prev_hand: Option<Hand> = None;
    let mut cshu: u32 = 0;
    for chr in ngram.chars() {
        let chr = chr as char;
        let (r, c) = char_to_key[&chr];
        let (hand, lk) = if c <= COL_HALF - 1 {
            (Hand::Left, LKey { row: r, lcol: c })
        } else {
            (Hand::Right, LKey { row: r, lcol: COL_COUNT - 1 - c })
        };

        let shift;
        if prev_hand.is_some_and(|ph| ph == hand) {
            shift = 0;
        } else {
            shift = cshu;
            cshu = 0;
            (other_prev_lk, prev_lk) = (prev_lk, other_prev_lk);
        }

        if let Some(plk) = prev_lk {
            let inward = lk.lcol > plk.lcol;
            let rightward = inward == (hand == Hand::Left);
            let horiz_abs = match (plk.row, lk.row) {
                (0, 0) | (1, 1) | (2, 2) => 0,
                (0, 1) | (1, 0) => 1,
                (0, 2) | (2, 0) => 3,
                (1, 2) | (2, 1) => 2,
                (_, _) => unreachable!(),
            };
            let upward = lk.row < plk.row;
            let rotate_inward = match hand {
                Hand::Left => upward == rightward,
                Hand::Right => upward != rightward,
            };
            let stretch = match hand {
                Hand::Left => !rotate_inward,
                Hand::Right => rotate_inward,
            };
            let stagger_score = match (stretch, rotate_inward) {
                (true, true) => -horiz_abs,
                (true, false) => 0,
                (false, true) => -horiz_abs,
                (false, false) => 0, // TODO: or horiz_abs?
            };
            // let rotate_score = if rotate_inward { -1 } else { 0 };

            finger_score += (count * (3 * finger_score_delta(lk.row, plk.row, lk.lcol, plk.lcol) + stagger_score)) >> shift;
        }

        prev_lk = Some(lk);
        prev_hand = Some(hand);
        cshu += 1;
        if cshu >= 3 {
            cshu_score -= count * (1 << (cshu - 3));
        }
    }

    (finger_score, cshu_score)
}

fn balance_score(
    letters: &[(String, u64)],
    char_to_key: &HashMap<char, (usize, usize)>,
) -> i64 {
    let mut left_sum: i64 = 0;
    let mut right_sum: i64 = 0;
    for &(ref g, count) in letters {
        let chr = g.chars().next().unwrap();
        let (_, c) = char_to_key[&chr];
        if c <= COL_HALF - 1 {
            left_sum += count as i64;
        } else {
            right_sum += count as i64;
        }
    }
    -(left_sum - right_sum).abs()
}

fn make_char_to_key(layout: &Layout) -> HashMap<char, (usize, usize)> {
    let mut char_to_key = HashMap::new();
    for (r, row) in layout.iter().enumerate() {
        for (c, &chr) in row.iter().enumerate() {
            let chr = chr as char;
            char_to_key.insert(chr, (r, c));
        }
    }
    char_to_key
}

fn layout_score(ngrams: &Ngrams, layout: &Layout, print_details: bool) -> i64 {
    for row in layout {
        for window in row.windows(3) {
            if window == &[0x41, 0x4E, 0x54] { // forbidden word
                return 0;
            }
        }
    }

    let char_to_key = make_char_to_key(layout);

    let mut ss = 0;
    for &(ref igram, count) in &ngrams[1] {
        ss += strength_score(igram, count, &char_to_key);
    }
    ss *= 6;

    let mut fs = 0;
    let mut hs = 0;
    for igrams in &ngrams[2..] {
        for &(ref igram, count) in igrams {
            let (fs_delta, hs_delta) = movement_score(igram, count, &char_to_key);
            fs += fs_delta;
            hs += hs_delta;
        }
    }
    fs *= 2;
    hs *= 18;

    let bs = 25 * balance_score(&ngrams[1], &char_to_key);

    if print_details {
        let format = num_format::CustomFormat::builder()
            .grouping(num_format::Grouping::Standard)
            .separator("_")
            .build().unwrap();

        print!("ss = ");
        io::stdout().write_formatted(&ss, &format).unwrap();
        print!("\n");
        print!("fs = ");
        io::stdout().write_formatted(&fs, &format).unwrap();
        print!("\n");
        print!("hs = ");
        io::stdout().write_formatted(&hs, &format).unwrap();
        print!("\n");
        print!("bs = ");
        io::stdout().write_formatted(&bs, &format).unwrap();
        print!("\n");
    }
    fs + hs + ss + bs
}

fn random_swap(
    layout: &Layout,
    n: &Uniform<usize>,
    swappable: &[(usize, usize)],
) -> Layout {
    let mut rng = thread_rng();

    // TODO: take this as a parameter for better perf
    let mut layout = layout.clone();

    let mut keys = Vec::new();
    let mut chars = Vec::new();

    let num_keys = n.sample(&mut rng);
    for &(r, c) in swappable.choose_multiple(&mut rng, num_keys) {
        keys.push((r, c));
        chars.push(layout[r][c]);
    }
    chars.shuffle(&mut rng);
    for (&(r, c), &chr) in keys.iter().zip(&chars) {
        layout[r][c] = chr;
    }

    layout
}

fn print_layout(layout: &Layout) {
    for row in layout {
        for (i, &chr) in row.iter().enumerate() {
            if i >= COL_HALF - 1 && i <= COL_HALF + 1 {
                print!(" ");
            }
            print!("{}", chr as char);
        }
        print!("\n");
    }
}

fn search(
    ngrams: &Ngrams,
    start_score: i64,
    start_layout: Layout,
    max_attempts: u64,
    swap_n: &Uniform<usize>,
    swappable: &[(usize, usize)],
    quiet: bool,
) -> (u64, Layout, i64) {  // (attempts, best layout, best score)
    let format = num_format::CustomFormat::builder()
        .grouping(num_format::Grouping::Standard)
        .separator("_")
        .build().unwrap();

    let mut best_score = start_score;
    let mut best_layout = start_layout;

    if !quiet {
        io::stdout().write_formatted(&best_score, &format).unwrap();
        print!("\n");
    }

    for i in 0..max_attempts {
        if PLEASE_STOP.load(Ordering::Acquire) {
            return (i, best_layout, best_score);
        }

        let layout = random_swap(&best_layout, swap_n, swappable);

        let score = layout_score(ngrams, &layout, false);
        if score > best_score {
            best_score = score;
            best_layout = layout;
            if !quiet {
                io::stdout().write_formatted(&score, &format).unwrap();
                print!("\n");
            }
        }
    }

    (max_attempts, best_layout, best_score)
}

#[derive(Debug)]
enum SearchType {
    Walk(i64),
    Peek(i64),
    Disturb(i64),
}

fn search_all(
    ngrams: &Ngrams,
    start_score: i64,
    start_layout: &Layout,
    max_attempts: &[SearchType],
    swap_n: &Uniform<usize>,
    swappable: &[(usize, usize)],
    quiet: bool,
) -> (u64, Layout, i64) {  // (attempts, best layout, best_score)
    use SearchType::*;
    let format = num_format::CustomFormat::builder()
        .grouping(num_format::Grouping::Standard)
        .separator("_")
        .build().unwrap();

    if max_attempts.len() == 1 {
        if let Walk(ma) = max_attempts[0] {
            assert!(ma > 0, "last max_attempts is negative or zero, which is stupid");
            return search(
                ngrams, start_score, start_layout.clone(), ma as u64, swap_n, swappable, quiet);
        } else {
            panic!("last max_attempts is Peek(_), which is stupid");
        }
    }

    if let Disturb(ma) = max_attempts[0] {
        let mut disturbed_layout = start_layout.clone();
        let new_layout = random_swap(&disturbed_layout, &Uniform::new(ma as usize, ma as usize + 1), swappable);
        disturbed_layout = new_layout;
        return search_all(
            ngrams,
            layout_score(ngrams, &disturbed_layout, false),
            &disturbed_layout,
            &max_attempts[1..],
            swap_n,
            swappable,
            quiet,
        );
    }

    let mut total_attempts = 0;
    let mut best_score = start_score;
    let mut best_layout = start_layout.clone();

    let ma = match max_attempts[0] { Walk(ma) => ma, Peek(ma) => ma, Disturb(_) => panic!() };
    if ma > 0 {
        for _ in 0..ma {
            if PLEASE_STOP.load(Ordering::Acquire) {
                break;
            }

            let (attempts, layout, score) = match max_attempts[0] {
                Walk(_) => search_all(ngrams, best_score, &best_layout, &max_attempts[1..], swap_n, swappable, quiet),
                Peek(_) => search_all(ngrams, start_score, &start_layout, &max_attempts[1..], swap_n, swappable, quiet),
                _ => panic!(),
            };
            total_attempts += attempts;
            if score > best_score {
                best_score = score;
                best_layout = layout;
            }

            for _ in 1..max_attempts.len() {
                print!("<");
            }
            print!("\n");
        }
    } else {
        match max_attempts[0] {
            Walk(_) => panic!(),
            Disturb(_) => panic!(),
            _ => (),
        }
        crossbeam::scope(|scope| {
            let mut children = Vec::new();
            for _ in 0..ma.abs() {
                children.push(scope.spawn(|_| {
                    search_all(
                        ngrams,
                        start_score,
                        start_layout,
                        &max_attempts[1..],
                        swap_n,
                        swappable,
                        true,
                    )
                }));
            }

            for child in children {
                let (attempts, layout, score) = child.join().unwrap();
                total_attempts += attempts;
                if score > best_score {
                    best_score = score;
                    best_layout = layout;
                }

                if !quiet {
                    io::stdout().write_formatted(&best_score, &format).unwrap();
                    print!("\n");
                }

                for _ in 1..max_attempts.len() {
                    print!("<");
                }
                print!("\n");
            }
        }).unwrap();

    }

    (total_attempts, best_layout, best_score)
}

fn sanitize_layout_char(chr: char) -> Option<char> {
    if let 'A'..='Z' | 'a'..='z' | '\'' | ',' | '.' | ';' = chr {
        Some(chr.to_ascii_uppercase())
    } else if chr.is_ascii_whitespace() {
        None
    } else {
        Some('-')
    }
}

fn read_layout() -> Vec<Vec<u8>> {
    let mut stdin = io::stdin();
    let mut layout = Vec::new();
    let mut b = vec![0];
    for _ in 0..=2 {
        let mut row = Vec::new();
        for _ in 0..=9 {
            let chr;
            loop {
                match stdin.read(&mut b) {
                    Ok(count) if count == 0 => {
                        eprintln!("Error: unexpected end of file");
                        exit(1);
                    },
                    Ok(_) => {
                        if let Some(c) = sanitize_layout_char(b[0] as char) {
                            chr = c;
                            break;
                        }
                    },
                    Err(e) if e.kind() == ErrorKind::Interrupted => (),
                    Err(e) => {
                        eprintln!("Error: {}", &e);
                        exit(1);
                    },
                }
            }
            row.push(chr as u8);
        }
        layout.push(row);
    }
    layout
}

static WHOLE_WORDS_ONLY: bool = false;

fn main() {
    let format = num_format::CustomFormat::builder()
        .grouping(num_format::Grouping::Standard)
        .separator("_")
        .build().unwrap();

    let mut args = env::args().skip(1);

    let cmd = args.next().unwrap();
    if cmd == "search" || cmd == "continue" {
        let nmax = args.next().unwrap().parse().unwrap();
        let swap_n_str = args.next().unwrap();
        let swap_n = if swap_n_str == "-" {
            Uniform::new(2usize, 8usize)
        } else {
            let n: usize = swap_n_str.parse().unwrap();
            Uniform::new(n, n + 1)
        };
        let ngrams = get_ngrams(nmax, WHOLE_WORDS_ONLY);

        let max_attempts: Vec<SearchType> = args.map(|x| {
            if x.ends_with(".") {
                SearchType::Peek(x[..x.len() - ".".len()].parse().unwrap())
            } else if x.ends_with("-") {
                SearchType::Disturb(x[..x.len() - "-".len()].parse().unwrap())
            } else {
                SearchType::Walk(x.parse().unwrap())
            }
        }).collect();

        let mut rng = rand::thread_rng();

        let mut start_layout = if cmd == "search" {
            let hyphens = vec!['-' as u8; COL_COUNT];
            vec![hyphens.iter().map(|&c| c as u8).collect(); 3]
        } else {
            read_layout()
        };

        let mut swappable = Vec::new();
        let mut every: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZ',.;".chars().collect();
        assert_eq!(every.len(), 30);
        for row in &start_layout {
            for &c in row {
                if let Some(i) = every.iter().position(|&c2| c2 == c as char) {
                    every.swap_remove(i);
                }
            }
        }
        every.shuffle(&mut rng);
        for r in 0..3 {
            for c in 0..COL_COUNT {
                if start_layout[r][c] as char == '-' {
                    swappable.push((r, c));
                    start_layout[r][c] = every.pop().unwrap() as u8;
                }
            }
        }

        if swappable.is_empty() {
            for r in 0..3 {
                for c in 0..COL_COUNT {
                    swappable.push((r, c));
                }
            }
        }

        let start_score = layout_score(&ngrams, &start_layout, false);

        if cmd == "continue" {
            println!("Continuing from this layout:");
        } else {
            println!("Starting from this layout:");
        }
        print_layout(&start_layout);
        if cmd == "continue" {
            io::stdout().write_formatted(&start_score, &format).unwrap();
            print!("\n");
        }
        println!();

        flag::register(SIGINT, PLEASE_STOP.clone()).unwrap();
        flag::register(SIGTERM, PLEASE_STOP.clone()).unwrap();

        let (attempts, best_layout, best_score) = search_all(
            &ngrams, start_score, &start_layout, &max_attempts, &swap_n, &swappable, false
        );
        println!();
        print_layout(&best_layout);
        layout_score(&ngrams, &best_layout, true);
        io::stdout().write_formatted(&best_score, &format).unwrap();
        print!("\n");
        println!("attempts: {} / {:?}", attempts, max_attempts);
        println!("n <= {}", nmax);
    } else if cmd == "score" {
        let nmax = args.next().unwrap().parse().unwrap();
        let ngrams = get_ngrams(nmax, WHOLE_WORDS_ONLY);

        let layout = read_layout();

        let score = layout_score(&ngrams, &layout, true);
        io::stdout().write_formatted(&score, &format).unwrap();
        print!("\n");
        println!("n <= {}", nmax);
    } else if cmd == "one-hand" {
        let nmax = args.next().unwrap().parse().unwrap();
        let ngrams = get_ngrams(nmax, true);

        let layout = read_layout();
        let char_to_key = make_char_to_key(&layout);

        for (i, igrams) in ngrams.iter().enumerate().skip(1) {
            println!("{}-gram", i);
            for &(ref igram, count) in igrams {
                if count == 0 {
                    continue;
                }
                let mut left = false;
                let mut right = false;
                for c in igram.chars() {
                    if char_to_key[&c].1 < COL_HALF {
                        left = true;
                    } else {
                        right = true;
                    }
                }
                if !(left && right) {
                    println!("{}", igram);
                }
            }
        }
    } else {
        panic!("Unknown subcommand '{}'", cmd);
    }
}
