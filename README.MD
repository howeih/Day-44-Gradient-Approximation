Day 44: Gradient Approximation
<br>
If you had to vote, what would you consider as the most important debugging tool ever? Without hesitation, gradient approximation would be my first choice.
![Alt text](gradient_approximation.png?raw=true "gradient_approximation")
<br>
Either was it neural network training or any other multivariate function optimization, it was very difficult due to bugs in derivatives. And there always were some bugs.
<br>
In my experience, it took 30 minutes to implement the task, and it took 6–12 more hours to make it work. Approximate gradient was super useful to identify the spots to be fixed.
<br>
It’s hard to describe how easy the life is with TensorFlow or Theano.
<br>
run:

```
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
```

result:

```
function: x= -1 grad=-3.999999999997339
function: x= 0 grad=2.000000000000335
function: x= 1 grad=7.999999999994678
x :[0.0, 0.0, 0.0] shape=[3], strides=[1], layout=C | F (0x3)
function2 x= [0, 0, 0] grad=[0, 1, 0.9997000199988333]
x :[0.0, 0.0, 1.0] shape=[3], strides=[1], layout=C | F (0x3)
function2 x= [0, 0, 1] grad=[1.8414709864744694, 1.5402023125343778, 0.9998301368663309]
x :[0.0, 1.0, 1.0] shape=[3], strides=[1], layout=C | F (0x3)
function2 x= [0, 1, 1] grad=[3.5597528177966353, 3.2583123242657663, 1.8762973599195654]
x :[1.0, 1.0, 1.0] shape=[3], strides=[1], layout=C | F (0x3)
function2 x= [1, 1, 1] grad=[8.23052709604255, 7.928619548440707, 7.086439873917882]
```
