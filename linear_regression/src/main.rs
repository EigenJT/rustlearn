fn main() {
    println!("Hello, world!");
    let f = 3u32;
    let g = 2u32;
    let hello = 5u32;
    println!("{}",hello);
    some_random_fucking_function(f);
    multiply(f,g);
    complicated(f,g,hello);

}

fn some_random_fucking_function(x: u32) {
    println!("{}",x);
}

fn multiply(x: u32, y: u32) {
    println!("{}",x*y);
}

fn complicated(x: u32, y: u32, z: u32) {
    let h = x*y*2;
    let e = x.pow(2);
    let l = 4;
    let o = y.pow(z);
    println!("{}",h*e*l*o);

}