use regex::{Regex, Captures};
use std::collections::{HashSet, HashMap};
use lazy_static::lazy_static;
use std::collections::hash_map::{RandomState, DefaultHasher};
use std::hash::{BuildHasher};
use std::fmt::Debug;
use std::iter;

pub struct Model<T> {
    vector_length: usize,
    corpus_size: usize,
    document_weights: Vec<Vec<f64>>,
    term_frequencies: Vec<Vec<usize>>,
    document_frequency: Vec<u64>,
    dictionary: HashSet<String>,
    index: HashMap<String, usize>,
    documents: Vec<T>,
    capture: fn(cap: &Captures) -> String,
    processing_regex: Regex,
}

pub trait Document {
    fn get_data(&self) -> String;
}

impl Document for String {
    fn get_data(&self) -> String {
        self.clone()
    }
}


impl<T> Model<T> where T: Document + Debug + Clone {
    pub fn preprocess(doc: &T, func: fn(cap: &Captures) -> String, processing_regex: Regex) -> Vec<String> {
        String::from(processing_regex.replace_all(&doc.get_data(), func).clone())
            .split(" ").clone()
            .map(|data| String::from(data).to_lowercase())
            .filter(|data| data != "" && data != " ")
            .collect::<Vec<_>>()
    }

    pub fn build_query_weights(&self, query_vec: &[i32]) -> Vec<f64> {
        query_vec.iter().enumerate().map(|(i, term)| {
            if *term == 0 { return 0f64; }
            Model::<T>::calc_query_tf(term) * Model::<T>::calc_query_idf(
                self.documents.len(),
                self.document_frequency[i],
            )
        }).collect::<Vec<f64>>()
    }

    pub fn search(&self, query: String) -> Vec<(T, f64)> {
        let preprocessed = Model::preprocess(
            &query,
            self.capture,
            self.processing_regex.clone(),
        );

        let mut query_vec = vec![0; self.vector_length];

        preprocessed.iter().for_each(|item| {
            if self.index.contains_key(item) {
                query_vec[*self.index.get(item).unwrap()] += 1
            }
        });


        //calc query weight
        let query_weight = self.build_query_weights(&query_vec);

        //calculate sim
        let mut vec = self.document_weights.iter().enumerate().map(|(i, doc)| {
            let res = (Model::<T>::sim(&query_weight, doc), i);
            res
        }).collect::<Vec<(f64, usize)>>();

        vec.sort_by(|f, b| b.0.partial_cmp(&f.0).unwrap());

        vec.iter().map(|item| { (self.documents[item.1].clone(), item.0) }).collect::<Vec<_>>()
    }

    pub fn calc_idf(term_df: u64, total_items: usize) -> f64 {
        (total_items as f64 / term_df as f64).log10()
    }

    pub fn calc_tf_idf(term_frequency: usize, idf: f64) -> f64 {
        if term_frequency == 0 { return 0f64; }
        (1f64 + (term_frequency as f64).log10()) * idf
    }

    pub fn calc_query_tf(term: &i32) -> f64 {
        (1.0 + (*term as f64).log10())
    }

    pub fn calc_query_idf(num_docs: usize, doc_freq: u64) -> f64 {
        (num_docs as f64 / doc_freq as f64).log10()
    }

    pub fn euclidean_len(v: &[f64]) -> f64 {
        let mut result = 0.0;
        for item in v {
            result += item.powi(2);
        }
        result.sqrt()
    }

    pub fn sim(query: &[f64], doc: &[f64]) -> f64 {
        let q_len = Model::<T>::euclidean_len(&query);
        let d_len = Model::<T>::euclidean_len(&doc);

        query.iter().enumerate().map(|(i, term)| (term / q_len) * (doc[i] / d_len))
            .sum::<f64>()
    }

    pub fn construct(documents: Vec<T>, processing_capture: fn(cap: &Captures) -> String, processing_regex: Regex) -> Self {
        let processed = documents.iter().map(|doc| {
            Model::preprocess(doc, processing_capture, processing_regex.clone())
        }).collect::<Vec<_>>();

        let mut dictionary = processed.clone().drain(..)
            .flatten()
            .collect::<HashSet<_>>();

        let num_docs = documents.len();
        let vector_len = dictionary.len();

        let mut document_frequency = vec![0; vector_len];
        let mut term_frequencies: Vec<Vec<usize>> = iter::repeat_with(|| vec![0; vector_len])
            .take(num_docs)
            .collect();

        let mut index = dictionary.iter().enumerate().map(
            |(i, item)| {
                (item.clone(), i)
            }
        ).collect::<HashMap<_, _>>();


        // built term frequency and document frequency
        for i in 0..num_docs {
            for term in processed[i].iter() {
                let term_index = index.get(term).unwrap().clone();
                if term_frequencies[i][term_index] == 0 {
                    document_frequency[term_index] += 1;
                }
                term_frequencies[i][term_index] += 1;
            }
        }


        // calc document weights
        let document_weights = term_frequencies.iter().map(|document| {
            document.iter().enumerate().map(|(i, tf)| {
                let idf = Model::<T>::calc_idf(
                    document_frequency[i],
                    documents.len(),
                );
                Model::<T>::calc_tf_idf(*tf, idf)
            }).collect::<Vec<_>>()
        }).collect::<Vec<Vec<f64>>>();

        Self {
            vector_length: vector_len,
            corpus_size: documents.len(),
            document_weights,
            term_frequencies,
            document_frequency,
            dictionary,
            index,
            documents,
            capture: processing_capture,
            processing_regex,
        }
    }
}