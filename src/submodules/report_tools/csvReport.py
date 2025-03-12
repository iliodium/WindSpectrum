import pandas as pd

accuracy = 5

n_arr = np.array([ox.round(accuracy),
                  data['cx'].round(accuracy),
                  data['cy'].round(accuracy),
                  data['cmz'].round(accuracy)
                  ]).T
df1 = pd.DataFrame(n_arr)

df1 = pd.DataFrame([ox.round(3).tolist(),
                    data['cx'].round(3).tolist(),
                    data['cy'].round(3).tolist(),
                    data['cmz'].round(3).tolist()
                    ])

df1 = pd.DataFrame([[1,2,3,4,5,6,7,8]])
df2 = pd.DataFrame([[1,2,3,4,5,6,7,8]])
df3 = pd.DataFrame([[1,2,3,4,5,6,7,8]])
df4 = pd.DataFrame([[1,2,3,4,5,6,7,8]])

df1 = df1.append(df2)
df1 = df1.append(df3)
df1 = df1.append(df4)

df1.to_csv('Интегрирование\\cx_cy_cmz_115_0_sum.csv', index=False, header=False, sep=',')