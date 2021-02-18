# vector-space-model
implementation of the vector space model for search  

based on [Scoring, term weighting and the vector space model](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html) over at Standfords npl dep

# Use: 
```Rust
// Implement Document trait for some type 
impl Document for String {
    fn get_data(&self) -> String {
        self.clone()
    }
}



fn main(){
    //Define a capture for regex, alternative is to just have the capture return a empty string 
    let capture = |cap: &Captures| {
        if cap.get(2).is_some() {
            return format!("{} {}", &cap[3], &cap[4]);
        }
        String::from(" ")
    };
    let data: Vec<String> = ... // what ever kind of input data as long as its a Vec<T> where T is bounded by Document
    let processing_regex:Regex = Regex::new(r"(x|\.)|((\d+)(mm))|([^A-Za-z0-9])").unwrap();
    let model = Model::construct(data, capture, processing_regex);
    let res = model.search(String::from("12"));
    
    println!("{:#?}",res)
    
    //Output will be on the form (Document, score)
}

```
