import matplotlib.pyplot as plt
import plotly_express as px
import wordcloud


def pie_plot(df, col_name):
    fig = px.pie(df.label.value_counts(), labels='index', values=col_name, color=col_name,
         color_discrete_sequence=["#871fff", "#ffa78c"])
    fig.show()


def histogram_plot(df, col_name):
    fig = px.histogram(df, x=col_name, color=col_name, color_discrete_sequence=["#871fff", "#ffa78c"])
    fig.show()


def show_word_cloud(df, col_name, title):
    text = ' '.join(df[col_name].astype(str).tolist())
    stopwords = set(wordcloud.STOPWORDS)
    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords, background_color='#ffa78c', width=3000,
                                        height=2000).generate(text)
    plt.figure(figsize=(15, 15), frameon=True)
    plt.imshow(fig_wordcloud)
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()

