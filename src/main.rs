#[macro_use(array)]
extern crate ndarray;
use ndarray::Array1;
use ndarray::ArrayViewMut;
use ndarray::Dim;

fn gradient<T>(fun: T, x: f64, delta: f64) -> f64
where
    T: Fn(f64) -> f64,
{
    let mut xi = x + delta;
    let mut grad = fun(xi);
    xi = x - delta;
    grad -= fun(xi);
    grad / (2. * delta)
}

fn gradient_ndarray<T>(
    fun: T,
    x: &mut ArrayViewMut<f64, Dim<[usize; 1]>>,
    delta: f64,
) -> Array1<f64>
where
    T: Fn(f64, f64, f64) -> f64,
{
    println!("x :{:?}", x);
    let mut y = [x[0], x[1], x[2]];
    let mut grad = Array1::<f64>::zeros(x.raw_dim());
    for (i, t) in x.iter().enumerate() {
        y[i] = *t + delta;
        grad[i] = fun(y[0], y[1], y[2]);
        y[i] = *t - delta;
        grad[i] -= fun(y[0], y[1], y[2]);
    }
    grad.iter().map(|x| x / (2. * delta)).collect()
}

fn function(x: f64) -> f64 {
    3. * x.powi(2) + 2. * x + 1.
}

fn function2(x: f64, y: f64, z: f64) -> f64 {
    x * z.sin() + y * z.cos() + z * (x + y).exp()
}

fn main() {
    for x in vec![-1., 0., 1.] {
        let res = gradient(function, x, 1e-4);
        println!("function: x= {} grad={}", x, res);
    }
    let mut a = array![[0., 0., 0.], [0., 0., 1.], [0., 1., 1.], [1., 1., 1.]];
    for mut x in a.outer_iter_mut() {
        let res = gradient_ndarray(function2, &mut x, 1e-4);
        println!("function2 x= {} grad={}", x, res);
    }
}
