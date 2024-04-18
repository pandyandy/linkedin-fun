import streamlit as st
import pandas as pd
import base64
import jwt
import vertexai
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

from google.oauth2 import service_account
from wordcloud import WordCloud
from vertexai.generative_models import GenerativeModel

PROJECT = "keboola-ai"
LOCATION = "us-central1"
MODEL_NAME = "gemini-1.5-pro-preview-0409"

CREDENTIALS = service_account.Credentials.from_service_account_info(
    jwt.decode(st.secrets["encoded_token"], 'keboola', algorithms=["HS256"])
)

# Calculate delta for metrics
def calculate_delta(current, overall):
    if overall == 0:
        return "N/A" if current == 0 else "∞"
    return f"{(current - overall) / overall * 100:.2f}%"

# Gemini
def generate(content):
    vertexai.init(project=PROJECT, location=LOCATION, credentials=CREDENTIALS)
    model = GenerativeModel(MODEL_NAME)

    config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    
    responses = model.generate_content(
        contents=content,
        generation_config=config,
        stream=True,
    )
    return "".join(response.text for response in responses)

# Logos
keboola_logo = "/data/in/files/282503_logo.png"
#qr_code = "/data/in/files/1112988135_qr_code.png"
keboola_gemini = "/data/in/files/282498_keboola_gemini.png"

# Load the data
data = pd.read_csv('/data/in/tables/linkedin_posts_categorized.csv')
keywords = pd.read_csv('/data/in/tables/keywords_grouped.csv')

# Data prep
data.rename(columns={'result_value': 'category', 'postedAtTimestamp': 'date'}, inplace=True)
data['date'] = pd.to_datetime(data['date'].astype(int) / 1000, unit='s')
data = data[data['authorFullName'] != 'Liviu Gherman']
allowed_categories = ["Educational", "Promotional", "Networking", "News and Updates", "Inspirational"]
data = data[data['category'].isin(allowed_categories) & (data['text'].str.len() >= 3)]

# Sidebar
#st.sidebar.markdown("""
#<div style="text-align: center;">
#    <h1>GCP Data Cloud Live</h1>
#    <br><p>Scan the QR code to see yourself on the dashboard:</p>
#</div>
#""", unsafe_allow_html=True)
#qr_html = f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{base64.b64encode(open(qr_code, "rb").read()).decode()}" style="width: 200px;"></div>'
#st.sidebar.markdown(f'{qr_html}', unsafe_allow_html=True)
#st.sidebar.markdown('<div style="text-align: center"><br><br><br>Get in touch with Keboola: <a href="https://bit.ly/cxo-summit-2024">https://bit.ly/cxo-summit-2024</a>#replace</div>', unsafe_allow_html=True)

# Title and Filters
keboola_logo_html = f'<div style="display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{base64.b64encode(open(keboola_logo, "rb").read()).decode()}" style="width: 150px; margin-left: -10px;"></div>'
st.markdown(f"{keboola_logo_html}", unsafe_allow_html=True)

st.title('LinkedIn Profiler')

author_list = data['authorFullName'].unique().tolist()
category_list = data['category'].unique().tolist()

selected_author = st.selectbox('Select an Author', ['All'] + author_list)

# Apply Filters
if selected_author != 'All':
    data_filtered = data[data['authorFullName'] == selected_author]
else:
    data_filtered = data

st.markdown("####")
# Show table
st.dataframe(data_filtered[['authorFullName',
                            'category', 
                            'commentsCount', 
                            'likesCount', 
                            'text']], 
             column_config={'authorFullName': 'Author', 
                            'category': 'Category',
                            'commentsCount': '# of Comments',
                            'likesCount': '# of Likes',
                            'text': 'Post'
                            },
            use_container_width=True, hide_index=True)


# Category Distribution
fig_category_distribution = px.pie(
    data_frame=data_filtered,
    names='category',
    title='Category Distribution',
)
st.plotly_chart(fig_category_distribution, use_container_width=True)

# Wordcloud
keywords_filtered = keywords[keywords['authorFullName'] == selected_author] if selected_author != 'All' else keywords[keywords['cnt'] > 3]
word_freq = keywords_filtered.set_index('keyword')['cnt'].to_dict()
colormap = mcolors.ListedColormap(['#0069c8', '#85c9fe', '#ff2a2b', '#feaaab', '#2bb19d'])

title_text = "Keyword Frequency"
if selected_author != 'All':
    title_text += f" for {selected_author}"

st.markdown(f"<br>**{title_text}**", unsafe_allow_html=True)

wordcloud = WordCloud(width=800, height=400, background_color=None, mode="RGBA", colormap=colormap).generate_from_frequencies(word_freq)
wordcloud_array = wordcloud.to_array()

plt.figure(figsize=(10, 5), frameon=False)
plt.imshow(wordcloud_array, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

with st.expander("More data:"):
    # Metrics
    total_posts = len(data_filtered)
    avg_likes = data_filtered['likesCount'].mean() if total_posts > 0 else 0
    avg_comments = data_filtered['commentsCount'].mean() if total_posts > 0 else 0

    total_posts_all = len(data)
    avg_likes_all = data['likesCount'].mean() if total_posts > 0 else 0
    avg_comments_all = data['commentsCount'].mean() if total_posts > 0 else 0

    _, col1, col2, col3, _ = st.columns(5)
    col1.metric("**📨 # of Posts**", total_posts)
    col2.metric("**👍🏻 Avg Likes**", f"{avg_likes:.2f}", delta=calculate_delta(avg_likes, avg_likes_all))
    col3.metric("**💬 Avg Comments**", f"{avg_comments:.2f}", delta=calculate_delta(avg_comments, avg_comments_all))



    # Activity Heatmap by Day and Hour
    data_filtered['weekday'] = data_filtered['date'].dt.day_name()
    data_filtered['hour'] = data_filtered['date'].dt.hour

    heatmap_data = data_filtered.groupby(['weekday', 'hour']).size().reset_index(name='posts_count')

    fig_heatmap = px.density_heatmap(
        data_frame=heatmap_data,
        x='hour',
        y='weekday',
        z='posts_count',
        nbinsx=24,
        category_orders={"weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
        title='Activity Heatmap by Day and Hour',
        labels={'posts_count': 'Posts', 'hour': 'Hour of the Day', 'weekday': 'Day of the Week'},
        height=400
    )
    fig_heatmap.update_xaxes(tickmode='linear', tick0=0, dtick=1)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    

# Gemini content analysis
gemini_html = f'<div style="display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{base64.b64encode(open(keboola_gemini, "rb").read()).decode()}" style="width: 60px; margin-top: 30px;"></div>'
st.markdown(f"{gemini_html}", unsafe_allow_html=True)

data_gemini = data_filtered.loc[:, ['category', 'text', 'likesCount', 'commentsCount', 'date']]
json_data = data_gemini.to_json(orient='records')

prompt_normal = f"""
You are given a data in JSON format that contains the author's LinkedIn posts in different categories and engagement metrics for each post, including the number of likes and comments, and the date of posting.

Analyze the data provided to identify patterns and insights that can inform the person's content strategy.

Expected outcome should be CONCISE and include:
– Summary of Findings: A brief report summarizing key insights and patterns from the data, including any correlations or anomalies found.
– Actionable Recommendations: Specific, data-driven suggestions for improving the content strategy to increase overall engagement.

Data: 
{json_data}
"""

prompt_fun = f"""
You are given a data in JSON format that contains the author's LinkedIn posts in different categories and engagement metrics for each post, including the number of likes and comments, and the date of posting.

Analyze the data provided to identify patterns and insights that can inform the person's content strategy – be satirical, you can make a little fun of it. You can use some emojis and slang words, too.

Expected outcome should be brief and include:
– Summary of Findings: A concise report summarizing key insights and patterns from the data, including any correlations or anomalies found.
– Actionable Recommendations: Specific, data-driven suggestions for improving the content strategy to increase overall engagement.

Data: 
{json_data}
"""

st.markdown("""
<div style="text-align: left;">
    <h4>Analyze the content strategy with Gemini</h4>
</div>
""", unsafe_allow_html=True)

if selected_author != 'All':
    st.markdown("""
    Choose your style:
    <br>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    if col1.button("Be serious 👨🏻‍💼", use_container_width=True):
        with st.spinner('Analyzing your posts... Please wait 👀'):
            generated_text = generate(prompt_normal)
        st.write(generated_text)

    if col2.button("Make it fun 🕺🏻", use_container_width=True):
        with st.spinner('Analyzing your posts... Please wait 👀'):
            generated_text = generate(prompt_fun)
        st.write(generated_text)
else:
    st.info("Select a specific author to analyze the content strategy with Gemini.")