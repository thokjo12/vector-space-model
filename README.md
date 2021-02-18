# vector-space-model
implementation of the vector space model for search  

based on [Scoring, term weighting and the vector space model](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html) over at Standfords npl dep

#Use: 
```Rust
// Implement Document trait for some type 
impl Document for String {
    fn get_data(&self) -> String {
        self.clone()
    }
}



fn main(){
    //Define a capture for regex alternative is to just have the capture return a empty string 
    let capture = |cap: &Captures| {
        if cap.get(2).is_some() {
            return format!("{} {}", &cap[3], &cap[4]);
        }
        String::from(" ")
    };
    let data = ... // what ever kind of input data as long as it implements Document
    let model = Model::construct(data, capture);
    let res = model.search(String::from("12"),capture);
    
    println!("{:?}",res)
}

```
