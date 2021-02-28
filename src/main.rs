use rand::prelude::*;
use rand::seq::index::sample;
use std::collections::HashMap;
use std::fs::File;
use std::io::{prelude::*, BufReader};

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

static KEY_TO_STRENGTH: &[&[u64]] = &[
    &[0, 1, 2, 2, 2, 2, 1, 0],
    &[3, 5, 8, 6, 6, 8, 5, 3],
    &[5, 7, 8, 8, 8, 8, 7, 5],
    &[3, 1, 4, 7, 7, 4, 1, 3],
];

fn strength_score(
    ngram: &str,
    count: u64,
    char_to_key: &HashMap<char, (usize, usize)>
) -> u64 {
    let mut score = 0;
    for chr in ngram.chars() {
        let key = char_to_key[&chr];
        let strength = KEY_TO_STRENGTH[key.0][key.1];
        score += strength * count;
    }
    score
}

fn layout_score(ngrams: &Ngrams, layout: &Layout) -> u64 {
    let mut score = 0;

    let mut char_to_key = HashMap::new();
    for (r, row) in layout.iter().enumerate() {
        for (c, &chr) in row.iter().enumerate() {
            let chr = chr as char;
            char_to_key.insert(chr, (r, c));
        }
    }

    for igrams in &ngrams[2..] {
        for &(ref igram, count) in igrams {
            score += strength_score(igram, count, &char_to_key);
        }
    }
    score
}

fn random_swap(layout: &Layout) -> Layout {
    let mut rng = thread_rng();
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
    start_score: u64,
    start_layout: Layout,
    max_attempts: &[u64],
) -> (u64, Layout) {  // (attempts, best layout)
    assert_eq!(max_attempts.len(), 1);
    let mut attempts = 0;
    let mut best_score = start_score;
    let mut best_layout = start_layout;

    for i in 0..max_attempts[0] {
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
            print_layout(&layout);
            println!("{}", score);
            println!();
            best_score = score;
            best_layout = layout;
        }
    }

    (max_attempts[0], best_layout)
}

fn main() {
    let ngrams = get_ngrams(2);
    println!("{:?}", search(&ngrams, 0, vec![
        b"...QJZ..".to_vec(),
        b"ABCDEFGH".to_vec(),
        b"IKLMNOPR".to_vec(),
        b"STUVWXY'".to_vec(),
    ], &[1000]));
}
