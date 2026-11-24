// two chat requests (json files) -> rendered prompts, common prefix, the text around the first difference
use dsx_core::server::{convert_chat_request, render_prompt};
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mut prompts = vec![];
    for f in &a {
        let body = std::fs::read(f).unwrap();
        let (conv, _, _) = convert_chat_request(&body, "m").unwrap_or_else(|e| panic!("{f}: {e:?}"));
        let r = render_prompt(&conv, "<image>");
        prompts.push(format!("{r:?}"));
    }
    for (i, p) in prompts.iter().enumerate() { println!("--- {} ---\n{p}", a[i]); }
}
