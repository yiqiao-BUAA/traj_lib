run the following command in the parent directory of `traj_lib` to test the model:
```bash
python -m traj_lib.main --model testmodel --dataset a --metrics accuracy
```
dataloader output format:
```python
data[dict]
user_id :       [batch_size] List[int]
POI_id :        [batch_size, seq_len] [numpy.array[int]]
time_sample :   [batch_size, seq_len] [numpy.array[int]]
mask ：         [batch_size, seq_len] [numpy.array[int]]
y_POI_id :      [batch_size] [numpy.array[int]]
y_time_sample : [batch_size] [numpy.array[int]]