use regex::{Regex, Captures};
use std::collections::{HashSet, HashMap};
use std::fmt::Debug;
use std::iter;

pub struct Model<T> {
    pub vector_length: usize,
    pub document_weights: Vec<Vec<f64>>,
    pub term_frequencies: Vec<Vec<usize>>,
    pub document_frequency: Vec<u64>,
    pub dictionary: HashSet<String>,
    pub index: HashMap<String, usize>,
    pub documents: Vec<T>,
    pub capture: fn(cap: &Captures) -> String,
    pub processing_regex: Regex,
    pub queued_for_indexing: Vec<T>,
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
    pub fn index() {}

    pub fn insert_docs(&mut self, docs: Vec<T>) {
        self.queued_for_indexing.extend(docs);
    }

    pub fn preprocess(doc: &T, func: &fn(cap: &Captures) -> String, processing_regex: &Regex) -> Vec<String> {
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
            &self.capture,
            &self.processing_regex.clone(),
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
        1.0 + (*term as f64).log10()
    }

    pub fn calc_query_idf(num_docs: usize, doc_freq: u64) -> f64 {
        (num_docs as f64 / doc_freq as f64).log10()
    }

    pub fn euclidean_len(v: &[f64]) -> f64 {
        v.iter().fold(0.0, |acc, elm| acc + elm.powi(2)).sqrt()
    }

    pub fn sim(query: &[f64], doc: &[f64]) -> f64 {



        let q_len = Model::<T>::euclidean_len(&query);
        let d_len = Model::<T>::euclidean_len(&doc);

        query.iter().enumerate().map(|(i, term)| (term / q_len) * (doc[i] / d_len))
            .sum::<f64>()
    }

    pub fn update_index(&mut self) { // TODO refactor and extract processing into separate functions for reuse in construct and update_index.
        if self.queued_for_indexing.is_empty() {
            println!("No documents queued");
            return;
        }

        let capture = &self.capture;
        let reg = &self.processing_regex;
        let num_docs = self.queued_for_indexing.len();

        self.documents.extend(self.queued_for_indexing.clone());
        let doc_count = self.queued_for_indexing.len();
        let processed = self.queued_for_indexing
            .drain(..)
            .map(|doc| Model::<T>::preprocess(&doc, capture, reg))
            .collect::<Vec<_>>();

        let tmp_dict = processed.clone()
            .drain(..)
            .flatten()
            .collect::<HashSet<_>>();


        // TODO swap dict, index to use indexmap: https://docs.rs/indexmap/1.6.2/indexmap/
        self.dictionary.extend(tmp_dict);

        let vector_length = self.dictionary.len();

        let mut document_frequency = vec![0; vector_length];
        let mut term_frequencies: Vec<Vec<usize>> = iter::repeat_with(|| vec![0; vector_length])
            .take(num_docs)
            .collect();


        self.index = self.dictionary
            .clone()
            .iter()
            .enumerate()
            .map(|(i, item)| { (item.clone(), i) })
            .collect::<HashMap<_, _>>();

        for i in 0..doc_count{
            for term in processed[i].iter(){
                let term_index = self.index.get(term).unwrap().clone();
                if term_frequencies[i][term_index] == 0{
                    document_frequency[term_index] += 1;
                }
                term_frequencies[i][term_index] += 1;
            }
        }

        self.document_frequency.iter().enumerate().map(|(i,df)| document_frequency[i]+df).collect::<Vec<_>>();

        // reconstruct the model.
        //extend self vars with the above.

    }

    pub fn calculate_document_weights(&mut self) {
        self.document_weights = self.term_frequencies.iter().map(|document| {
            document.iter().enumerate().map(|(i, tf)| {
                let idf = Model::<T>::calc_idf(
                    self.document_frequency[i],
                    self.documents.len(),
                );
                Model::<T>::calc_tf_idf(*tf, idf)
            }).collect::<Vec<_>>()
        }).collect::<Vec<Vec<f64>>>();
    }

    pub fn construct(documents: Vec<T>, processing_capture: fn(cap: &Captures) -> String, processing_regex: Regex) -> Self {
        let processed = documents.iter().map(|doc| {
            Model::preprocess(doc, &processing_capture, &processing_regex.clone())
        }).collect::<Vec<_>>();

        let dictionary = processed.clone().drain(..)
            .flatten()
            .collect::<HashSet<_>>();

        let num_docs = documents.len();
        let vector_len = dictionary.len();

        let mut document_frequency = vec![0; vector_len];
        let mut term_frequencies: Vec<Vec<usize>> = iter::repeat_with(|| vec![0; vector_len])
            .take(num_docs)
            .collect();

        let index = dictionary.iter().enumerate().map(
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

        let mut model = Self {
            vector_length: vector_len,
            capture: processing_capture,
            queued_for_indexing: vec![],
            document_weights:vec![],
            term_frequencies,
            document_frequency,
            dictionary,
            index,
            documents,
            processing_regex,
        };

        model.calculate_document_weights();
        model
    }
}