# AML_2019_Group20
Gradient Descent on Schaffer function No2
# Part 1 

## Gradient Descent in Layman Terms 

*'Basically, gradient descent helps us get the most accurate prediction based on some data.'*

Well, this was not very informative, so let us explain a bit more. Let's say you have a list of height and weight, and let's say you graph it, it would probably look something like this: 

![figure 1 - Observations of weight and height](https://cdn-images-1.medium.com/max/1200/1*Zv0P0I3MT6DO1mH0zaD-2w.png) 

Now let's say there is a local guessing competition where the person to guess someone's weight correctly, given their height gets a price. Besides using your eyes, the list above would be very useful. 

So, based on the graph of your data above, you could probably make a reasonable good predictions. if only we had a line on the graph that showed the *trend* of the data. With such a line, given someones weight, you could just find that weight on the x-axis go up until you hit the trend line, and then see what is the corresponding height on the y-axis! ($y_i = \alpha + x\beta_i + \epsilon_i$) 

But how would you find the *perfect* line? That is where gradient descent is your friend! 

It does this by minimizing some form of error function (let's say RSS - residual sum of squares), which is simply the sum of the squared differences between our dots (the actual observations) and our line. We get a smaller RSS by changing where our line is on the graph, and we stop changing the position of the graph when the line is closest to the majority of our dots - which is beautifully visualized below! 

![Animation of Gradient Descent](https://media.giphy.com/media/O9rcZVmRcEGqI/giphy.gif)

Taking this further we can graph each different line's parameters on something called a cost curve. Using GD we can get to the bottom if our curve, where we have the corresponding lowest RSS.

### Why is GD important? 

As we have seen Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost). Gradient descent is best used when the parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched for by an optimization algorithm. in ML we aim to find the combined set of parameters which yield the best possible results, just like gradient descent! Just like the animation above, our initial start is pretty bad, and with a very high rate of error, but after iterating over new possible solutions (walking down the hill) we find better and better combinations, until we have found an optimum solution where we are not able to reduce the error anymore 

### Modifications to GD 

**Mini Batch Gradient Descent**. Instead of using the full data set or a single instance, this algorithm, as the name indicates, computes gradients on small random sets of instances called mini batches. This algorithm is able to reduce the noise from the Stochastic and still be more efficient than full-batch. Basically, it splits the training data set into small batches ranging from 10 to 1,000 examples, chosen at random.

**Stochastic Gradient Descent** is an alternative to vanilla gradient descent. It uses a single example (batch of 1) per each learning step (new iterations with corresponding new RSS/loss results). Constant update allows us to have a detailed rate of improvement. It is much faster since it has few data to manipulate at every iteration, however, it can return gradients with unstable error rate. Therefore, as it “jumps around” (stochastic behavior) when the algorithm stops instead of finding the optimal fit it ends up obtaining only good results. Due to its stochastic nature, the algorithm is much less regular than the previous one. 


# Plain vanilla gradient descent on a function with two variables
The chosen function is the Schaffer function No2:

𝑓(𝑥,𝑦)=0.5+𝑠𝑖𝑛2(𝑥2−𝑦2)−0.5(1+0.001(𝑥2+𝑦2))2
 
It has two variables x and y, many local minima, one global maxima and one global minima.

Global minima:
𝑓(𝑥,𝑦)=0,𝑎𝑡(𝑥,𝑦)=(0,0)
 
The function is usually evaluated on the square x ∈ [-100, 100] and y ∈ [-100, 100] for the further visualisation below.

![figure 2 - Visualisation of the Schaffer function No2]

To illustrate plain vanilla gradient decsent on Schaffer function No2, a class <b>gd_pv</b> is created. It is instantiated with with a loss function, and two functions that return the gradient of the loss function at any given value of x and y. 
- fn_loss
- fn_grad_x
- fn_grad_y

The class contains a method <b>step</b>, which minimises the cost of the given loss function and where the actual gradient descent takes place. It takes initial values for x and y and updates them based on steps taken via the learning rate. 

Then, it outputs the most minimum values of x and y that reaches the stop condition (precision stop), which means that when the absolute difference between our old and updated x and y values are greater than a precision value, the algorithm should stop. 

The method will also print out the minumum of the loss function, the minimum values of x and y, as well as the number of steps or descents it took to reach that value.

It takes which takes following parameters, which must be provided by the user:
- x_init: an initial guess for the value of x that minimises the function
- y_init: an initial guess for the value of y that minimises the function
- n_iter: maximum given number of iterations for the algorythm
- precision: a precision that determines the stop of the stepwise descent 
- l_r: learning rate or step size

Next, we define the argument functions for the gradient descent class.

- fn_loss: takes in x and y arguments, returns out loss function - Schaffer function No2
- fn_grad_x: takes in x and y arguments, returns differentiated gradient function for any given value of x
- fn_grad_x: takes in x and y arguments, returns differentiated gradient function for any given value of y

Now, we can apply plain vanilla gradient descent by instantiating the gd_pv class with the given loss and gradient functions. With given parameters, gradient descent with learning rate of 0.001 was able to minimise the Schaffer function No2 in 1540979 steps. By minimising the function, the plain vanilla gradient descent finds the corresponding values of x and y determining the global minimum.

*Number of steps: 1540979*

*The minimum of the loss function: 0.003126615752*

*Global minimum occurs at:
x = 1.7723150369215577
y = 0.002523625545385108*

To evaluate the changes that occur in the implementation of the gradient descent with different step sizes, we take the same parameters and gradually decrease from 0.01 to 0.0001. Given the complexity of the Schaffer No2 function the number of iterations is set to 10000000. Apart from the step size, the remaining ouput should not change.

---------------------------------------------------------
Learning Rate is  0.01, Number of steps: 154096
---------------------------------------------------------
Learning Rate is  0.003, Number of steps: 513658
---------------------------------------------------------
Learning Rate is  0.002, Number of steps: 770488
---------------------------------------------------------
Learning Rate is  0.001, Number of steps: 1540979
---------------------------------------------------------
Learning Rate is  0.0001, Number of steps: 15409814
---------------------------------------------------------

From the output of the loop, it generally can be that the smaller is the learning rate, the larger is the number of steps required for the gradient descent to minimise the function.

**Momentum gradient descent**. For vanilla gradient descent, we use plain derivative as gradient to update the values of x and y. In momentum, instead of using dW and db independently for each iteration, we take the exponentially weighted averages of x and y, with alpa is another hyperparameter called momentum and ranges from 0 to 1. It sets the weight between the average of previous values and the current value to calculate the new weighted average. After calculating exponentially weighted averages, we will update our parameters. The class for momentum method is created below. Alpha must be predefined by the user.

**Nesterov accelerated gradient**. Another method that is closely related to Momentum method is Nesterov Accelerated Gradient. Momentum method, the gradient was computed using current parameters of nu, whereas in Nesterov Accelerated Gradient, we apply the velocity mu to the parameters nu to compute interim parameters. We then compute the gradient using those interim parameters.

- Plain Vanilla GD: 1540979 steps
- Momentum GD: 6044 steps
- Nesterev GD: 5150 steps

Comparing three different Gradient Descent methods results, it can clearly been seen that more complex algorithms that backed up by more sophisticated update rules at each iteration t, like Momentum abd  Nesterev GDs, manage to minimise the cost function more efficiently taking the less number of steps. In Leiman terms, while plain vanilla GD simply jumps to the next value of calculated x and y based on the differentiation, more complex GDs jump in a more randomised way. Momentum and Nesterev GD are quite similar to each other, however, for Schaffer No2 function Nesterev GD is the most efficient. 
