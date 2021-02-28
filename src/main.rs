use num_format::WriteFormatted;
use rand::prelude::*;
use rand::seq::index::sample;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, prelude::*, BufReader};

type Layout = Vec<Vec<u8>>;
type Ngrams = Vec<Vec<(String, u64)>>;

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
    &[0, 1, 2, 2, 2, 2, 1, 0],
    &[3, 5, 8, 6, 6, 8, 5, 3],
    &[5, 7, 8, 8, 8, 8, 7, 5],
    &[3, 1, 4, 7, 7, 4, 1, 3],
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
    let mut row1 = false;
    let mut row3 = false;
    let mut prev_col = -1;
    for chr in ngram.chars() {
        let chr = chr as char;
        let (r, c) = char_to_key[&chr];
        match r {
            0 => return 0,
            1 => row1 = true,
            3 => row3 = true,
            _ => (),
        }
        if (prev_col <= 3 && c as isize <= prev_col)
                || (prev_col >= 4 && c as isize >= prev_col) {
            if c as isize == prev_col {
                return -(count as i64 * ngram.len() as i64);
            } else {
                return 0;
            }
        }
        prev_col = c as isize;
    }
    if row1 && row3 {
        return 0;
    } else {
        return count as i64 * ngram.len() as i64;
    }
}

fn layout_score(ngrams: &Ngrams, layout: &Layout) -> i64 {
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
    for igrams in &ngrams[2..] {
        for &(ref igram, count) in igrams {
            irs += inward_roll_score(igram, count, &char_to_key);
        }
    }
    15 * irs + ss
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
    max_attempts: &[u64],
) -> (u64, Layout) {  // (attempts, best layout)
    let format = num_format::CustomFormat::builder()
        .grouping(num_format::Grouping::Standard)
        .separator("_")
        .build().unwrap();

    assert_eq!(max_attempts.len(), 1);
    let mut best_score = start_score;
    let mut best_layout = start_layout;
    let mut failed = 0;

    for _ in 0..max_attempts[0] {
        let mut layout;
        loop {
            layout = random_swap(&best_layout);
            if layout[0].iter().filter(|&&chr| chr == '.' as u8).count() == 5
                    && !layout[0].contains(&('\'' as u8)) {
                break;
            }
        }

        let score = layout_score(ngrams, &layout);
        if score > best_score {
            println!();
            println!("failed = {}", failed);
            println!();
            print_layout(&layout);
            io::stdout().write_formatted(&score, &format).unwrap();
            print!("\n");
            best_score = score;
            best_layout = layout;
            failed = 0;
        } else {
            failed += 1;
        }
    }

    (max_attempts[0], best_layout)
}

fn main() {
    let format = num_format::CustomFormat::builder()
        .grouping(num_format::Grouping::Standard)
        .separator("_")
        .build().unwrap();

    let ngrams = get_ngrams(4);

    let max_attempts: Vec<u64> =
        env::args().skip(1).map(|x| x.parse().unwrap()).collect();

    let mut rng = rand::thread_rng();

    let mut not_qxz = b"ABCDEFGHIJKLMNOPRSTUVWY'".clone();
    assert_eq!(not_qxz.len(), 26 + 1 - 3);
    not_qxz.shuffle(&mut rng);
    let mut qxz = b"QXZ.....".clone();
    qxz.shuffle(&mut rng);

    let (attempts, best_layout) = search(&ngrams, 0, vec![
        qxz.to_vec(),
        not_qxz[0..8].to_vec(),
        not_qxz[8..16].to_vec(),
        not_qxz[16..24].to_vec(),
    ], &max_attempts);
    println!();
    print_layout(&best_layout);
    let best_score = layout_score(&ngrams, &best_layout);
    io::stdout().write_formatted(&best_score, &format).unwrap();
    print!("\n");
    println!("attempts: {} / {:?}", attempts, max_attempts);
}
