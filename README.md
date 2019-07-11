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


