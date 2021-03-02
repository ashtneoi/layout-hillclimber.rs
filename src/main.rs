use lazy_static::lazy_static;
use num_format::WriteFormatted;
use rand::prelude::*;
use rand::seq::index::sample;
use signal_hook::consts::signal::{SIGINT, SIGTERM};
use signal_hook::flag;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, prelude::*, BufReader};
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

static KEY_TO_STRENGTH: &[&[i64]] = &[
    &[-6, -4, -3, -4, -4, -3, -4, -6],
    &[1, 7, 8, 6, 6, 8, 7, 1],
    &[6, 9, 10, 9, 9, 10, 9, 6],
    &[3, 1, 1, 8, 8, 1, 1, 3],
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

fn inward_roll_score(
    ngram: &str,
    count: u64,
    char_to_key: &HashMap<char, (usize, usize)>,
) -> i64 {
    let score_mult = count as i64 * ngram.len() as i64;

    let mut inward = true;
    let mut prev_col = -1;
    for chr in ngram.chars() {
        let chr = chr as char;
        let (_, c) = char_to_key[&chr];
        if (prev_col <= 3 && c as isize <= prev_col)
                || (prev_col >= 4 && c as isize >= prev_col) {
            if c as isize == prev_col {
                // CSFU; not acceptable
                return 0;
            } else {
                // outward
                inward = false;
            }
        }
        prev_col = c as isize;
    }

    if inward {
        2 * score_mult
    } else {
        score_mult
    }
}

fn row_score(
    ngram: &str,
    count: u64,
    char_to_key: &HashMap<char, (usize, usize)>,
) -> i64 {
    let score_mult = count as i64 * ngram.len() as i64;

    let mut rows_in_hand = [
        [false, false, false, false],
        [false, false, false, false],
    ];

    for chr in ngram.chars() {
        let chr = chr as char;
        let (r, c) = char_to_key[&chr];
        rows_in_hand[if c <= 3 { 0 } else { 1 }][r] = true;
    }

    let mut score = 0;
    for rows in &rows_in_hand {
        score += match rows {
            &[true, _, _, _] => 0,
            &[false, true, true, true] => 0,
            | &[false, true, true, false]
            | &[false, false, true, true] => score_mult,
            _ => 2 * score_mult,
        };
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
        if c <= 3 {
            left_sum += count as i64;
        } else {
            right_sum += count as i64;
        }
    }
    -(left_sum - right_sum).abs()
}

fn layout_score(ngrams: &Ngrams, layout: &Layout, print_details: bool) -> i64 {
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
    let mut irs = 0;
    let mut rs = 0;
    for igrams in &ngrams[2..] {
        for &(ref igram, count) in igrams {
            irs += inward_roll_score(igram, count, &char_to_key);
            rs += row_score(igram, count, &char_to_key);
        }
    }
    let bs = balance_score(&ngrams[1], &char_to_key);
    if print_details {
        let format = num_format::CustomFormat::builder()
            .grouping(num_format::Grouping::Standard)
            .separator("_")
            .build().unwrap();

        print!("irs = ");
        io::stdout().write_formatted(&irs, &format).unwrap();
        print!("\n");
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
    2 * irs + ss + 8000 * bs
}

fn random_swap(layout: &Layout) -> Layout {
    let mut rng = thread_rng();

    // TODO: take this as a parameter for better perf
    let mut layout = layout.clone();

    let mut keys = Vec::new();
    let mut chars = Vec::new();
    // TODO: Rng::gen_range isn't optimal if we're calling it in a loop.
    let num_keys = rng.gen_range(2..=7);
    for key_num in sample(&mut rng, 32, num_keys) {
        let r = key_num >> 3;
        let c = key_num & 0x7;
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
        for &chr in row {
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

        let mut layout;
        loop {
            layout = random_swap(&best_layout);
            if layout[0].iter().filter(|&&chr| chr == '.' as u8).count() == 5
                    && !layout[0].contains(&('\'' as u8)) {
                break;
            }
        }

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
    let nmax = args.next().unwrap().parse().unwrap();
    let ngrams = get_ngrams(nmax);

    let max_attempts: Vec<i64> =
        args.map(|x| x.parse().unwrap()).collect();

    let mut rng = rand::thread_rng();

    let mut not_qxz = b"ABCDEFGHIJKLMNOPRSTUVWY'".clone();
    assert_eq!(not_qxz.len(), 26 + 1 - 3);
    not_qxz.shuffle(&mut rng);
    let mut qxz = b"QXZ.....".clone();
    qxz.shuffle(&mut rng);

    flag::register(SIGINT, PLEASE_STOP.clone()).unwrap();
    flag::register(SIGTERM, PLEASE_STOP.clone()).unwrap();

    let (attempts, best_layout, best_score) = search_all(&ngrams, 0, &vec![
        qxz.to_vec(),
        not_qxz[0..8].to_vec(),
        not_qxz[8..16].to_vec(),
        not_qxz[16..24].to_vec(),
    ], &max_attempts);
    println!();
    print_layout(&best_layout);
    io::stdout().write_formatted(&best_score, &format).unwrap();
    print!("\n");
    layout_score(&ngrams, &best_layout, true);
    println!("attempts: {} / {:?}", attempts, max_attempts);
    println!("n <= {}", nmax);
}
