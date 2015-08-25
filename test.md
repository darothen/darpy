# Test

This is just a blank test markdown file for checking if `ipymd` is working. It has a simple math equation:

$$\text{Area(circle)} = 2\pi r^2, \qquad \pi \approx 3.1415$$

a [link to google](http://www.google.com), and a test Python code cell:

```python
>>> a = 1; b = 2
>>> print("hello world!")
>>> print(a + b)
```

and a plot of a $\sin$ curve:

```python
>>> %matplotlib inline
...
>>> import matplotlib.pyplot as plt
>>> import seaborn as sns; sns.set_style("ticks")
...
>>> from numpy import linspace, sin, pi
...
>>> x = linspace(0, 2.*pi, 1000)
>>> y = sin(x)
...
>>> plt.plot(x, y, lw=2, color='k', label='$\sin(x)$')
>>> sns.despine(offset=10)
>>> plt.legend(loc='best')
>>> plt.ylim(-1.2, 1.2)
...
>>> ax = plt.gca()
>>> ax.spines['bottom'].set_position('zero')
```
