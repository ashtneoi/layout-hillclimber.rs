use std::collections::HashMap;
use std::fs::File;
use std::io::{prelude::*, BufReader};

type Layout = Vec<String>;
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
    char_to_key: HashMap<char, (usize, usize)>
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
    for igrams in &ngrams[2..] {
        for igram in igrams {
            println!("{:?}", igram);
        }
    }
    score
}

fn random_swap(layout: &Layout) -> Layout {
    layout.clone()
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
            if layout[0].matches('.').count() == 5
                    && !layout[0].contains('\'') {
                break;
            }
        }

        let score = layout_score(ngrams, &layout);
        if score > best_score {
            best_score = score;
            best_layout = layout;
        }
    }

    (attempts, best_layout)
}

fn main() {
    let ngrams = get_ngrams(2);
    println!("{:?}", search(&ngrams, 0, vec![
        "...QJZ..".to_string(),
        "ABCDEFGH".to_string(),
        "IKLMNOPR".to_string(),
        "STUVWXY'".to_string(),
    ], &[2]));
}
