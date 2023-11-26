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
static ROLL_TABLE: &'static [[[i64; 5]; 5]; 5] = &[
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

static COL_COUNT: isize = 10;
static COL_HALF: isize = COL_COUNT / 2;

static KEY_TO_STRENGTH: &[&[i64]] = &[
    &[6, 12, 14, 10, 4, 4, 10, 14, 12, 6],
    &[10, 16, 20, 16, 7, 7, 16, 20, 16, 10],
    &[0, 2, 2, 14, 5, 5, 14, 2, 2, 0],
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

fn roll_score_delta(
    r: isize, prev_r: isize,
    lc: isize, prev_lc: isize,
) -> i64 {
    ROLL_TABLE[(r - prev_r + 2) as usize][prev_lc as usize][lc as usize]
        - if (r - prev_r).abs() == 2 { 2 } else { 0 }
        + 2
}

fn roll_score(
    ngram: &str,
    count: u64,
    char_to_key: &HashMap<char, (usize, usize)>,
) -> i64 {
    let count = count as i64;
    let mut score: i64 = 0;

    let mut same_hand_length = 0;

    let mut prev_r: isize = -1;
    let mut prev_c: isize = -1;
    let mut other_prev_r: isize = -1;
    let mut other_prev_c: isize = -1;
    for chr in ngram.chars() {
        let chr = chr as char;
        let (r, c) = char_to_key[&chr];
        let r = r as isize;
        let c = c as isize;

        let prev_lc;
        let lc;
        if prev_c <= COL_HALF - 1 {
            if c >= COL_HALF { // hand swap
                same_hand_length = 0;
                (prev_r, other_prev_r) = (other_prev_r, prev_r);
                (prev_c, other_prev_c) = (other_prev_c, prev_c);
                prev_lc = COL_COUNT - 1 - prev_c;
                lc = COL_COUNT - 1 - c;
            } else {
                prev_lc = prev_c;
                lc = c;
            }
        } else {
            if c <= COL_HALF - 1 { // hand swap
                same_hand_length = 0;
                (prev_r, other_prev_r) = (other_prev_r, prev_r);
                (prev_c, other_prev_c) = (other_prev_c, prev_c);
                prev_lc = prev_c;
                lc = c;
            } else {
                prev_lc = COL_COUNT - 1 - prev_c;
                lc = COL_COUNT - 1 - c;
            }
        }
        same_hand_length += 1;

        if same_hand_length >= 3 {
            score -= 4 * (same_hand_length - 2) * count;
        }

        if prev_r != -1 {
            let shift = if same_hand_length == 0 { 1 } else { 0 };
            score += (count * roll_score_delta(r, prev_r, lc, prev_lc)) >> shift;
        }

        prev_r = r;
        prev_c = c;
    }

    score
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
        if c <= COL_HALF as usize - 1 {
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
    ss *= 5;
    let mut rs = 0;
    for igrams in &ngrams[2..] {
        for &(ref igram, count) in igrams {
            rs += roll_score(igram, count, &char_to_key);
        }
    }
    rs *= 1;
    let bs = 70 * balance_score(&ngrams[1], &char_to_key);
    if print_details {
        let format = num_format::CustomFormat::builder()
            .grouping(num_format::Grouping::Standard)
            .separator("_")
            .build().unwrap();

        print!("rs = ");
        io::stdout().write_formatted(&rs, &format).unwrap();
        print!("\n");
        print!("ss = ");
        io::stdout().write_formatted(&ss, &format).unwrap();
        print!("\n");
        print!("bs = ");
        io::stdout().write_formatted(&bs, &format).unwrap();
        print!("\n");
    }
    rs + ss + bs
}

fn random_swap(layout: &Layout) -> Layout {
    let mut rng = thread_rng();

    // TODO: take this as a parameter for better perf
    let mut layout = layout.clone();

    let mut keys = Vec::new();
    let mut chars = Vec::new();
    // TODO: Rng::gen_range isn't optimal if we're calling it in a loop.
    let num_keys = rng.gen_range(2..=7);
    for key_num in sample(&mut rng, 30, num_keys) {
        let r = key_num / 10;
        let c = key_num % 10;
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
            if i == row.len() / 2 {
                print!(" ");
            }
            print!("{}", chr as char);
        }
        print!("\n");
    }
}

// hill-climbing random walk
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

// random peek
fn search_all(
    ngrams: &Ngrams,
    start_score: i64,
    start_layout: &Layout, // seed layout?
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

fn read_layout() -> Vec<Vec<u8>> {
    let mut stdin = io::stdin();
    let mut layout = Vec::new();
    let mut b = vec![0];
    for _ in 0..=2 {
        let mut row = Vec::new();
        for _ in 0..=9 {
            let mut chr;
            loop {
                match stdin.read(&mut b) {
                    Ok(count) if count == 0 => {
                        eprintln!("Error: unexpected end of file");
                        exit(1);
                    },
                    Ok(_) => {
                        chr = b[0] as char;
                        if let 'A'..='Z' | 'a'..='z' | '\'' | '.' = chr {
                            chr.make_ascii_uppercase();
                            break;
                        } else if let '-' | '_' = chr {
                            chr = '.';
                            break;
                        } else if !chr.is_ascii_whitespace() {
                            chr = '.';
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
            let mut every = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ'...".clone();
            assert_eq!(every.len(), 30);
            every.shuffle(&mut rng);

            start_layout = vec![
                every[0..COL_COUNT as usize].to_vec(),
                every[COL_COUNT as usize..2*COL_COUNT as usize].to_vec(),
                every[2*COL_COUNT as usize..3*COL_COUNT as usize].to_vec(),
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
