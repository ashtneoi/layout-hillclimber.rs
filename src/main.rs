use lazy_static::lazy_static;
use num_format::WriteFormatted;
use rand::prelude::*;
use rand::seq::index::sample;
use signal_hook::consts::signal::{SIGINT, SIGTERM};
use signal_hook::flag;
use std::cmp::max;
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
        let fields: Vec<_> = line.splitn(3, '\t').collect();
        let kind = fields[0];
        let splats = fields[1];
        assert_eq!(kind, &format!("{}-gram", i));
        assert_eq!(splats, "*/*");
        line.clear();
        let mut igrams = Vec::new();
        while lines.read_line(&mut line).unwrap() > 0 {
            let fields: Vec<_> = line.splitn(3, '\t').collect();
            let igram = fields[0];
            if igram.ends_with("-gram") {
                // Don't clear line (it's the next header).
                break;
            }
            let count = fields[1];
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
    &[7, 12, 14, 9, 4, 4, 9, 14, 12, 7],
    &[12, 16, 20, 16, 7, 7, 16, 20, 16, 12],
    &[7, 8, 8, 14, 1, 1, 14, 8, 8, 7],
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
                continue;
            }
            prev_lc = prev_c;
            lc = c;
        } else {
            if c <= COL_HALF - 1 { // hand swap
                same_hand_length = 0;
                continue;
            }
            prev_lc = COL_COUNT - 1 - prev_c;
            lc = COL_COUNT - 1 - c;
        }
        let prev_lf = max(prev_lc, 3);
        let lf = max(lc, 3);
        same_hand_length += 1;

        if same_hand_length >= 4 {
            score -= (same_hand_length - 2) * count;
        }

        if lf == prev_lf {
            if r == prev_r {
                // nothing
            } else if lf == 0 {
                score -= 4 * count;
            } else if lf == 1 {
                score -= 2 * (r - prev_r).abs() * count;
            } else { // lf == 2 || lf == 3
                score -= (r - prev_r).abs() * count;
            }
        } else if prev_lf == 0 { // inward roll from pinky
            if lc == 4 { // lateral stretch
                if r == prev_r - 2 { // two rows upward
                    score -= 20 * count;
                } else if r == prev_r - 1 { // one row upward
                    score -= 10 * count;
                } else if r == prev_r + 1 { // one row downward
                    // nothing
                } else { // two rows downward
                    score -= 10 * count;
                }
            } else if r == prev_r + 1 { // downward
                score -= 3 * count;
            } else if r == prev_r - 1 { // upward
                score += count;
            } else if ... {
                ...
            }
        } else if prev_lf == 1 { // roll from ring
            if lf == 0 { // outward
                if r == prev_r + 1 { // downward
                    // nothing
                } else if r == prev_r - 1 { // upward
                    score -= 5 * count;
                } else { // same row
                    score += count;
                }
            } else if lf == 2 { // ring to middle
                if r == prev_r + 1 { // downward
                } else if r == prev_r - 1 { // upward
                } else { // same row
                }
            } else if lf == 3 { // ring to index
                if lc == 4 { // lateral stretch
                }
            }
        } else if prev_lf == 2 { // roll from middle
            if lf < 2 { // outward
            } else { // middle to index
                if lc == 4 { // lateral stretch
                    if r == prev_r - 2 { // two rows upward
                    } else if r == prev_r - 1 { // one row upward
                    } else if r == prev_r + 1 { // one row downward
                    } else { // two rows downward
                    }
                }
            }
        } else { // outward roll from index
            if prev_lc == 4 { // lateral stretch
                if lf == 0 { // index to pinky
                    if r == prev_r - 2 { // two rows upward
                    } else if r == prev_r - 1 { // one row upward
                    } else if r == prev_r + 1 { // one row downward
                    } else { // two rows downward
                    }
                } else { // index to middle or ring
                    if r == prev_r - 2 { // two rows upward
                    } else if r == prev_r - 1 { // one row upward
                    } else if r == prev_r + 1 { // one row downward
                    } else { // two rows downward
                    }
                }
            }
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
    let mut rs = 0;
    for igrams in &ngrams[2..] {
        for &(ref igram, count) in igrams {
            rs += roll_score(igram, count, &char_to_key);
        }
    }
    rs *= 5;
    let bs = 100 * balance_score(&ngrams[1], &char_to_key);
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

fn search(
    ngrams: &Ngrams,
    start_score: i64,
    start_layout: Layout,
    max_attempts: u64,
) -> (u64, Layout, i64) {  // (attempts, best layout, best score)
    let mut best_score = start_score;
    let mut best_layout = start_layout;

    for i in 0..max_attempts {
        if PLEASE_STOP.load(Ordering::Acquire) {
            return (i, best_layout, best_score);
        }

        let layout = random_swap(&best_layout);

        let score = layout_score(ngrams, &layout, false);
        if score > best_score {
            best_score = score;
            best_layout = layout;
        }
    }

    (max_attempts, best_layout, best_score)
}

fn search_all(
    ngrams: &Ngrams,
    start_score: i64,
    start_layout: &Layout, // seed layout?
    max_attempts: &[i64],
) -> (u64, Layout, i64) {  // (attempts, best layout, best_score)
    if max_attempts.len() == 1 {
        assert!(max_attempts[0] > 0);
        return search(
            ngrams, start_score, start_layout.clone(), max_attempts[0] as u64);
    }

    let mut total_attempts = 0;
    let mut best_score = start_score;
    let mut best_layout = start_layout.clone();

    if max_attempts[0] > 0 {
        for _ in 0..max_attempts[0] {
            if PLEASE_STOP.load(Ordering::Acquire) {
                break;
            }

            let (attempts, layout, score) = search_all(
                ngrams, start_score, &best_layout, &max_attempts[1..]);
            total_attempts += attempts;
            if score > best_score {
                best_score = score;
                best_layout = layout;
            }

            println!();
            for _ in 1..max_attempts.len() {
                print!("<");
            }
            print!("\n");
        }
    } else {
        crossbeam::scope(|scope| {
            let mut children = Vec::new();
            for _ in 0..max_attempts[0].abs() {
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

                println!();
                for _ in 1..max_attempts.len() {
                    print!("<");
                }
                print!("\n");
            }
        }).unwrap();
    }

    (total_attempts, best_layout, best_score)
}

fn main() {
    let format = num_format::CustomFormat::builder()
        .grouping(num_format::Grouping::Standard)
        .separator("_")
        .build().unwrap();

    let mut args = env::args().skip(1);

    let cmd = args.next().unwrap();
    if cmd == "search" {
        let nmax = args.next().unwrap().parse().unwrap();
        let ngrams = get_ngrams(nmax);

        let max_attempts: Vec<i64> =
            args.map(|x| x.parse().unwrap()).collect();

        let mut rng = rand::thread_rng();

        let mut every = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ'...".clone();
        assert_eq!(every.len(), 30);
        every.shuffle(&mut rng);

        flag::register(SIGINT, PLEASE_STOP.clone()).unwrap();
        flag::register(SIGTERM, PLEASE_STOP.clone()).unwrap();

        let (attempts, best_layout, best_score) = search_all(&ngrams, 0, &vec![
            every[0..COL_COUNT as usize].to_vec(),
            every[COL_COUNT as usize..2*COL_COUNT as usize].to_vec(),
            every[2*COL_COUNT as usize..3*COL_COUNT as usize].to_vec(),
        ], &max_attempts);
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

        print_layout(&layout);
        let score = layout_score(&ngrams, &layout, true);
        io::stdout().write_formatted(&score, &format).unwrap();
        print!("\n");
        println!("n <= {}", nmax);
    }
}
