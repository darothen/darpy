```python
%load_ext autoreload
```

```python
%autoreload
import xray
import marc_analysis

%matplotlib inline
import seaborn as sns
sns.set(style='ticks', context='talk')
```

```python
data = xray.open_dataset("/Users/daniel/Desktop/MARC_AIE"
                         "/F1850/arg_comp/arg_comp.cam2.h0.0008-01.nc")
```

```python
TS_zavg = (data['TS']
                .mean(dim='lon', keep_attrs=True)
                .squeeze())

line_plot = marc_analysis.vis.plot.line_plot
ax, lp = line_plot(TS_zavg, 'lat')
```

```python
pd = (data['TS']
         .mean('time', keep_attrs=True))

cmap_kwargs = marc_analysis.vis.infer_cmap_params(pd, levels=21, vmin=210, vmax=305,
                                                  cmap='spectral')
print(cmap_kwargs)
marc_analysis.vis.geo_plot(pd, projection='Robinson', **cmap_kwargs)
```

```python
%autoreload
import marc_analysis
marc_analysis.vis.plot2d(pd, func_kwargs={'projection': 'Robinson'})
```

```python
%autoreload
import marc_analysis

zd = (data['T']
         .mean('time', keep_attrs=True)
         .mean('lon'))

ax, zvp = marc_analysis.vertical_plot(zd, log_vert=True)

marc_analysis.plot2d(zd)
```

```python
a = set([1, 2])
a.pop()
```

```python
test == test
```

```python
import functools

def _default_func(func):

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        return func(*args, **kwargs)

    setattr(func, "default", True)
    return wrapped_func
```

```python

```
