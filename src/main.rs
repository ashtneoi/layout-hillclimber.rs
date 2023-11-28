use lazy_static::lazy_static;
use num_format::WriteFormatted;
use rand::prelude::*;
use rand::seq::index::sample;
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

fn get_ngrams(maxlen: usize) -> Ngrams {
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
        let splats = fields[2];
        assert_eq!(kind, &format!("{}-gram", i));
        assert_eq!(splats, &format!("{}/*", i));
        line.clear();
        let mut igrams = Vec::new();
        while lines.read_line(&mut line).unwrap() > 0 {
            let fields: Vec<_> = line.splitn(4, '\t').collect();
            let igram = fields[0];
            if igram.ends_with("-gram") {
                // Don't clear line (it's the next header).
                break;
            }
            let count = fields[2];
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
    &[9, 14, 20, 16, 8, 4, 16, 20, 14, 9],
    &[10, 16, 20, 18, 10, 10, 18, 20, 16, 10],
    &[0, 2, 2, 14, 4, 10, 16, 2, 2, 0],
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
    MOVEMENT_TABLE[(r - prev_r + 2) as usize][prev_lc as usize][lc as usize]
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
            finger_score += (count * finger_score_delta(lk.row, plk.row, lk.lcol, plk.lcol)) >> shift;
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

fn layout_score(ngrams: &Ngrams, layout: &Layout, print_details: bool) -> i64 {
    for row in layout {
        for window in row.windows(3) {
            if window == &[0x41, 0x4E, 0x54] { // forbidden word
                return 0;
            }
        }
    }

    let mut char_to_key = HashMap::new();
    for (r, row) in layout.iter().enumerate() {
        for (c, &chr) in row.iter().enumerate() {
            let chr = chr as char;
            char_to_key.insert(chr, (r, c));
        }
    }

    let mut ss = 0;
    for &(ref igram, count) in &ngrams[1] {
        ss += strength_score(igram, count, &char_to_key);
    }
    ss *= 25;

    let mut fs = 0;
    let mut hs = 0;
    for igrams in &ngrams[2..] {
        for &(ref igram, count) in igrams {
            let (fs_delta, hs_delta) = movement_score(igram, count, &char_to_key);
            fs += fs_delta;
            hs += hs_delta;
        }
    }
    fs *= 3;
    hs *= 2;

    let bs = 70 * balance_score(&ngrams[1], &char_to_key);

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

fn random_swap(layout: &Layout) -> Layout {
    let mut rng = thread_rng();

    // TODO: take this as a parameter for better perf
    let mut layout = layout.clone();

    let mut keys = Vec::new();
    let mut chars = Vec::new();
    // TODO: Rng::gen_range isn't optimal if we're calling it in a loop.
    if rng.gen_ratio(1, 3) {
        let num_keys = rng.gen_range(2..=5);
        for key_num in sample(&mut rng, 8, num_keys) {
            let r = 1;
            let c = if key_num < 4 { key_num } else { key_num + COL_COUNT - 8 };
            keys.push((r, c));
            chars.push(layout[r][c]);
        }
    } else {
        let num_keys = rng.gen_range(2..=7);
        for key_num in sample(&mut rng, 2 * COL_COUNT, num_keys) {
            let (r, c) = if key_num < COL_COUNT {
                (0, key_num)
            } else if key_num < 2 * COL_COUNT {
                (2, key_num - COL_COUNT)
            } else {
                (1, key_num - 2 * COL_COUNT + 4)
            };
            keys.push((r, c));
            chars.push(layout[r][c]);
        }
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
) -> (u64, Layout, i64) {  // (attempts, best layout, best score)
    let format = num_format::CustomFormat::builder()
        .grouping(num_format::Grouping::Standard)
        .separator("_")
        .build().unwrap();

    let mut best_score = start_score;
    let mut best_layout = start_layout;

    io::stdout().write_formatted(&best_score, &format).unwrap();
    print!("\n");

    for i in 0..max_attempts {
        if PLEASE_STOP.load(Ordering::Acquire) {
            return (i, best_layout, best_score);
        }

        let layout = random_swap(&best_layout);

        let score = layout_score(ngrams, &layout, false);
        if score > best_score {
            best_score = score;
            best_layout = layout;
            io::stdout().write_formatted(&score, &format).unwrap();
            print!("\n");
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
) -> (u64, Layout, i64) {  // (attempts, best layout, best_score)
    use SearchType::*;

    if max_attempts.len() == 1 {
        if let Walk(ma) = max_attempts[0] {
            assert!(ma > 0, "last max_attempts is negative or zero, which is stupid");
            return search(
                ngrams, start_score, start_layout.clone(), ma as u64);
        } else {
            panic!("last max_attempts is Peek(_), which is stupid");
        }
    }

    if let Disturb(ma) = max_attempts[0] {
        let mut disturbed_layout = start_layout.clone();
        for _ in 0..ma {
            let new_layout = random_swap(&disturbed_layout);
            disturbed_layout = new_layout;
        }
        return search_all(
            ngrams, layout_score(ngrams, &disturbed_layout, false), &disturbed_layout, &max_attempts[1..]);
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
                Walk(_) => search_all(ngrams, best_score, &best_layout, &max_attempts[1..]),
                Peek(_) => search_all(ngrams, start_score, &start_layout, &max_attempts[1..]),
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
                        ngrams, start_score, start_layout, &max_attempts[1..])
                }));
            }

            for child in children {
                let (attempts, layout, score) = child.join().unwrap();
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
        Some('.')
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

fn main() {
    let format = num_format::CustomFormat::builder()
        .grouping(num_format::Grouping::Standard)
        .separator("_")
        .build().unwrap();

    let mut args = env::args().skip(1);

    let cmd = args.next().unwrap();
    if cmd == "search" || cmd == "continue" {
        let nmax = args.next().unwrap().parse().unwrap();
        let home_row: Vec<char>;
        if cmd == "search" {
            home_row = args.next().unwrap().chars().filter_map(sanitize_layout_char).collect();
            assert_eq!(8, home_row.len());
        } else {
            home_row = Vec::new();
        }
        let ngrams = get_ngrams(nmax);

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

        let start_layout;
        if cmd == "search" {
            let every = "ABCDEFGHIJKLMNOPQRSTUVWXYZ',.;";
            assert_eq!(every.len(), 30);
            let mut outer_rows: Vec<char> = every.chars().filter(|c| !home_row.contains(c)).collect();
            outer_rows.shuffle(&mut rng);
            let mut middle_row = Vec::new();
            middle_row.extend_from_slice(&home_row[0..4]);
            middle_row.append(&mut outer_rows.split_off(outer_rows.len() - 2));
            middle_row.extend_from_slice(&home_row[4..8]);

            start_layout = vec![
                outer_rows[0..COL_COUNT].iter().map(|&c| c as u8).collect(),
                middle_row.iter().map(|&c| c as u8).collect(),
                outer_rows[COL_COUNT..2*COL_COUNT].iter().map(|&c| c as u8).collect(),
            ];
        } else {
            start_layout = read_layout();
        }

        let start_score = layout_score(&ngrams, &start_layout, false);

        if cmd == "continue" {
            println!("Continuing from this layout:");
            print_layout(&start_layout);
            io::stdout().write_formatted(&start_score, &format).unwrap();
            print!("\n");
            println!();
        }

        flag::register(SIGINT, PLEASE_STOP.clone()).unwrap();
        flag::register(SIGTERM, PLEASE_STOP.clone()).unwrap();

        let (attempts, best_layout, best_score) = search_all(&ngrams, start_score, &start_layout, &max_attempts);
        println!();
        print_layout(&best_layout);
        layout_score(&ngrams, &best_layout, true);
        io::stdout().write_formatted(&best_score, &format).unwrap();
        print!("\n");
        println!("attempts: {} / {:?}", attempts, max_attempts);
        println!("n <= {}", nmax);
    } else if cmd == "score" {
        let nmax = args.next().unwrap().parse().unwrap();
        let ngrams = get_ngrams(nmax);

        let layout = read_layout();

        print_layout(&layout);
        let score = layout_score(&ngrams, &layout, true);
        io::stdout().write_formatted(&score, &format).unwrap();
        print!("\n");
        println!("n <= {}", nmax);
    }
}
