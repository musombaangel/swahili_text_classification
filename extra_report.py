import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

data=pd.read_excel('extra_report/predictions.xlsx')

st.title('Performance Report')
st.caption('by Angel Musomba')
st.write('This is a report of the model\'s training and findings')

label_cols=['Actual_Value','Predicted_Value']
results=data[label_cols]

st.subheader('Results')
st.write('The table below shows the actual values and values predicted by the model.')
st.write(results)



#pie chart
st.subheader('Pie chart of the results')
st.write('The pie chart below shows the percentage proportion of the predicted values.')
counts=data['Predicted_Value'].value_counts()
labels=counts.index
values=counts.values
fig=go.Figure(data=[go.Pie(labels=labels, values=values, hoverinfo='label+percent', textinfo='value')])
st.plotly_chart(fig)


#confusion matrix
st.subheader('Confusion Matrix')
st.write('The following confusion matrix shows the distribution of predicted and actual values which reflects the performance of the model.')
conf_matrix=pd.crosstab(data['Actual_Value'],data['Predicted_Value'], colnames=['Predicted Value'], rownames=['Actual Value'])
st.write(conf_matrix)
st.write('Results:')
st.write('For purposes of this application, a \'positive\' is a value that is scam and a negative is one that is not. They are labelled as trust.')
st.write('True positives = 72 (The number of positive values correctly predicted)')
st.write('True negatives = 21 (The number of negative values correctly predicted)')
st.write('False positives = 72 (The number of negative values incorrectly predicted as positive.)')
st.write('False negatives = 0: The number of positive values incorrectly predicted as negative.')

#model accuracy
st.write('Model Accuracy: 92.66%')

#wordcloud
st.subheader('Wordcloud of the results')
st.write('The wordcloud below shows the most common words in the results.'
' The size of the word is proportional to the frequency of the word in the results.')

#access the wordcloud image(can be generated in reports/visualizations.ipynb)
with open('images/wordcloud.png','rb') as f:
    image=f.read()

st.image(image,use_container_width=True)

