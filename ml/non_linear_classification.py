import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('classification_data_14.csv')
df_C1 = df.loc[df.Class=='+']
df_C2 = df.loc[df.Class=='-']

x1_Max_C1 = df_C1.loc[:,'X1'].max()
x2_Max_C1 = df_C1.loc[:,'X2'].max()
x1_Max_C2 = df_C2.loc[:,'X1'].max()
x2_Max_C2 = df_C2.loc[:,'X2'].max()
x1_Min_C1 = df_C1.loc[:,'X1'].min()
x2_Min_C1 = df_C1.loc[:,'X2'].min()
x1_Min_C2 = df_C2.loc[:,'X1'].min()
x2_Min_C2 = df_C2.loc[:,'X2'].min()

down_Specific  = max(x2_Min_C1, x2_Min_C2)
up_Specific    = min(x2_Max_C1, x2_Max_C2)
left_Specific  = max(x1_Min_C1, x1_Min_C2)
right_Specific = min(x1_Max_C1, x1_Max_C2)

if x2_Min_C1 > x2_Min_C2 :
    outer_df = df_C2
else :
    outer_df = df_C1

left_Arr  = outer_df[outer_df['X1'] < left_Specific ].sort_values('X1', ascending=False)
right_Arr = outer_df[outer_df['X1'] > right_Specific].sort_values('X1', ascending=True )
down_Arr  = outer_df[outer_df['X2'] < down_Specific ].sort_values('X2', ascending=False)
up_Arr    = outer_df[outer_df['X2'] > up_Specific   ].sort_values('X2', ascending=True )

left_P  = left_Arr.iloc[0]
right_P = right_Arr.iloc[0]
up_P    = up_Arr.iloc[0]
down_P  = down_Arr.iloc[0]


left_Gen  = left_P['X1']
right_Gen = right_P['X1']
down_Gen  = down_P['X2']
up_Gen    = up_P['X2']

reset = 1

li = 0
ri = 0
ui = 0
di = 0

while (reset == 1):
    reset = 0
    if (not left_Gen < up_P['X1'] < right_Gen):
        reset = 1
        ui = ui + 1
        up_P = up_Arr.iloc[ui]
        up_Gen    = up_P['X2']

    if (not left_Gen < down_P['X1'] < right_Gen):
        reset = 1
        di = di + 1
        down_P = down_Arr.iloc[di]
        down_Gen  = down_P['X2']

    if (not down_Gen < left_P['X2'] < up_Gen):
        reset = 1
        li = li + 1
        left_P = left_Arr.iloc[li]
        left_Gen  = left_P['X1']

    if (not down_Gen < right_P['X2'] < up_Gen):
        reset = 1
        ri =  ri + 1
        right_P = right_Arr.iloc[ri]
        right_Gen = right_P['X1']

left_Opt  = (left_Specific  + left_Gen ) / 2
right_Opt = (right_Specific + right_Gen) / 2
up_Opt    = (up_Specific    + up_Gen   ) / 2
down_Opt  = (down_Specific  + down_Gen ) / 2

fig = px.scatter(df, x = 'X1', y = 'X2', color="Class")
fig.add_shape(
            type="rect",
            x0=left_Specific,
            y0=down_Specific,
            x1=right_Specific,
            y1=up_Specific,
            line=dict(
                color="RoyalBlue",
            ),
        )

fig.add_shape(
            type="rect",
            x0=left_Gen,
            y0=down_Gen,
            x1=right_Gen,
            y1=up_Gen,
            line=dict(
                color="Red",
            ),
        )

fig.add_shape(
            type="rect",
            x0=left_Opt,
            y0=down_Opt,
            x1=right_Opt,
            y1=up_Opt,
            line=dict(
                color="Green",
            ),
        )
fig.show()
