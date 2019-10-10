# PythonDeepLearning Repository

## Supervised Deep Learning
### CNNs 
1. Papers 
	1. CNNS
        1. [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
        2. [Introduction to Convolutional Neural Networks](https://www.semanticscholar.org/paper/Introduction-to-Convolutional-Neural-Networks-Wu/450ca19932fcef1ca6d0442cbf52fec38fb9d1e5)
        3. [The 9 Deep Learning Papers You Need To Know About (Understanding CNNs Part 3)](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)
    2. ReLU
        1. [Understanding Convolutional Neural Networks with a Mathematical Model](https://arxiv.org/pdf/1609.04112.pdf)
        2. [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf)
    3. Pooling
        1. [Evaluation of Pooling Operations in Convolutional Architectures for Object Recognition](http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf)
1.Resources
	1. CNN    
		1. [The Ultimate Guide to Convolutional Neural Networks (CNN)](https://www.superdatascience.com/blogs/the-ultimate-guide-to-convolutional-neural-networks-cnn)
		2. [Gimp Convolution Matrix Plug-in](https://docs.gimp.org/2.6/en/plug-in-convmatrix.html)
     
### RNNs
1. Papers
	1. [Learning Long-Term Dependencies with Gradient Descent is Difficult](http://ai.dinfo.unifi.it/paolo//ps/tnn-94-gradient.pdf)
	2. [On the Difficult of Training Recurrent Neural Networks](http://www.jmlr.org/proceedings/papers/v28/pascanu13.pdf)
	3. [LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069.pdf)
	4. [Long Short-Term Memory](http://bioinf.jku.at/publications/older/2604.pdf) - The original paper.
	5. [Visualizing and Understanding Recurrent Networks](https://arxiv.org/pdf/1506.02078.pdf)
	6. [LSTM: A Search Space Odyssey](https://arxiv.org/pdf/1503.04069.pdf)
2. Blogs
	1. (Understanding LSTM Networks)[http://colah.github.io/posts/2015-08-Understanding-LSTMs/]
	2. [Understanding LSTMs and its diagrams](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714)
	3. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
	4. [Hacker's guide to Neural Networks](http://karpathy.github.io/neuralnets/)
3. Notes 
	1. Vanishing Gradient
		1. Solutions:
			1. Weight Initialization
			2. Echo State Networks
			3. Long Short-Term Memory Networks (LSTMs)
	2. Exploding Gradient 
		1. Solutions: 
			1. Truncated Backpropagation
			2. Penalties
			3. Maximum Clipping
	3. LSTMs
		1. Theory: LSTMs have a memory cell which flows through "time" freely. In some cases things are added or updated, but otherwise flows relatively freely, thus removing the vanishing gradient problem.
		2. [LSTM Chain Image](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
			1. Vector Transfer: Any line moving through the architecture.
			2. Concatenation: 2 lines combining into one. Best way to imagine it is that lines are moving in parallel. 
			3. Pointwise Operations: "X", "+"
				1. Think of it as a valve
				2. Represented in literature formulas as: "f" "v" "o"

## Unsupervised Deep Learning
### Self Organizing Maps (SOM)
1. Blogs
    1. [AI Junkie SOM Example](http://www.ai-junkie.com/ann/som/som1.html)
2. K Means Cluster
	1. WCSS - Within-Cluster-Sum-of-Squares

				
### Blogs
1. Andrej Karpathy [karpathy.github.io](http://karpathy.github.io/) | [https://medium.com/@karpathy](https://medium.com/@karpathy)
### Udemy Courses
1. [Deep Learning A-Z™: Hands-On Artificial Neural Networks](https://www.udemy.com/course/deeplearning/)
2. [Machine Learning, Data Science and Deep Learning with Python](https://www.udemy.com/course/data-science-and-machine-learning-with-python-hands-on/)