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

st.set_page_config(layout='wide')

PROJECT = "keboola-ai"
LOCATION = "us-central1"
MODEL_NAME = "gemini-1.5-pro-preview-0409"

credentials = service_account.Credentials.from_service_account_info(
    jwt.decode(st.secrets["encoded_token"], 'keboola', algorithms=["HS256"])
)

# Logos
keboola_logo = "/data/in/files/1112932287_logo.png"
qr_code = "/data/in/files/1112988135_qr_code.png"
keboola_gemini = "/data/in/files/1112932549_keboola_gemini.png"

keboola_logo_html = f'<div style="display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{base64.b64encode(open(keboola_logo, "rb").read()).decode()}" style="width: 150px; margin-left: -10px;"></div>'
st.markdown(f"{keboola_logo_html}", unsafe_allow_html=True)

def calculate_delta(current, overall):
    if overall == 0:
        return "N/A" if current == 0 else "∞"
    return f"{(current - overall) / overall * 100:.2f}%"

def generate(content):
    vertexai.init(project=PROJECT, location=LOCATION, credentials=credentials)
    model = GenerativeModel(MODEL_NAME)

    config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    
    responses = model.generate_content(
        contents=f"""
You are given a data in JSON format that contains the author's social media posts in different categories and engagement metrics for each post, including the number of likes and comments.

Please analyze the data provided to identify patterns and insights that can inform the content strategy.

Expected outcome should be brief and include:
– Summary of Findings: A concise report summarizing key insights and patterns from the data, including any correlations or anomalies found.
– Actionable Recommendations: Specific, data-driven suggestions for improving the content strategy to increase overall engagement.

Data: 
{content}
""",
        generation_config=config,
        stream=True,
    )
    return "".join(response.text for response in responses)


# Load the data
data = pd.read_csv('/data/in/tables/linkedin_posts_categorized.csv')
keywords = pd.read_csv('/data/in/tables/keywords_grouped.csv')

# Data prep
data.rename(columns={'result_value': 'category', 'postedAtTimestamp': 'date'}, inplace=True)
data['date'] = pd.to_datetime(data['date'].astype(int) / 1000, unit='s')
allowed_categories = ["Educational", "Promotional", "Networking", "News and Updates", "Inspirational"]
data = data[data['category'].isin(allowed_categories) & (data['text'].str.len() >= 3)]

# Sidebar
st.sidebar.markdown("""
<div style="text-align: center;">
    <h1>GCP Data Cloud Live</h1>
    <br><p>Scan the QR code to see yourself on the dashboard:</p>
</div>
""", unsafe_allow_html=True)
qr_html = f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{base64.b64encode(open(qr_code, "rb").read()).decode()}" style="width: 200px;"></div>'
st.sidebar.markdown(f'{qr_html}', unsafe_allow_html=True)
st.sidebar.markdown('<div style="text-align: center"><br><br><br>Get in touch with Keboola: <a href="https://bit.ly/cxo-summit-2024">https://bit.ly/cxo-summit-2024</a>#replace</div>', unsafe_allow_html=True)

# Title and Filters
st.title('LinkedIn Profiler')

author_list = data['authorFullName'].unique().tolist()
category_list = data['category'].unique().tolist()

col1, col2 = st.columns(2)
with col1: 
    selected_author = st.selectbox('Select an Author', ['All'] + author_list)
with col2:
    selected_category = st.multiselect('Select Categories', category_list, default=category_list, 
                                       help = """
- Educational (sharing knowledge and insights)
- Promotional (products, services, or events)
- Networking (seeking connections, collaborations)
- News and Updates (company or industry news)
- Inspirational (motivational content)
""")

# Apply Filters
data_filtered_all = data[data['category'].isin(selected_category)]

if selected_author != 'All':
    data_filtered = data[(data['category'].isin(selected_category)) & (data['authorFullName'] == selected_author)]
else:
    data_filtered = data[data['category'].isin(selected_category)]

# Metrics
st.markdown("####")

total_posts = len(data_filtered)
avg_likes = data_filtered['likesCount'].mean() if total_posts > 0 else 0
avg_comments = data_filtered['commentsCount'].mean() if total_posts > 0 else 0

total_posts_all = len(data_filtered_all)
avg_likes_all = data_filtered_all['likesCount'].mean() if total_posts > 0 else 0
avg_comments_all = data_filtered_all['commentsCount'].mean() if total_posts > 0 else 0

_, col1, col2, col3, _ = st.columns(5)
col1.metric("**📨 # of Posts**", total_posts)
col2.metric("**👍🏻 Avg Likes**", f"{avg_likes:.2f}", delta=calculate_delta(avg_likes, avg_likes_all))
col3.metric("**💬 Avg Comments**", f"{avg_comments:.2f}", delta=calculate_delta(avg_comments, avg_comments_all))


col1, col2 = st.columns([4, 3], gap='medium')
with col1:
    # Author Engagement
    author_engagement_data = data_filtered.groupby('authorFullName').agg({'likesCount': 'mean'}).reset_index()
    author_engagement_data = author_engagement_data.sort_values(by='likesCount', ascending=False).head(10)

    fig_author_engagement = px.bar(
        data_frame=author_engagement_data,
        x='authorFullName',
        y='likesCount',
        title='Author Engagement',
        labels={'likesCount': 'Avg Likes per Post', 'authorFullName': 'Author'},
        text='likesCount'  
    )
    
    fig_author_engagement.update_traces(texttemplate='%{text:.2s}', textposition='inside')
    st.plotly_chart(fig_author_engagement, use_container_width=True)

with col2:
    # Category Distribution
    fig_category_distribution = px.pie(
        data_frame=data_filtered,
        names='category',
        title='Category Distribution',
    )
    st.plotly_chart(fig_category_distribution, use_container_width=True)

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
                            'text': 'Post',
                            'date': 'Date',
                            },
            use_container_width=True, hide_index=True)

# Post Activity Over Time
#data_2024 = data_filtered[(data_filtered['date'].dt.year == 2023) | (data_filtered['date'].dt.year == 2024)]
#data_2024['month_year'] = data_2024['date'].dt.to_period('M').dt.strftime('%Y-%m')

#fig_posts_over_time = px.bar(
#    data_frame=data_2024.groupby(['month_year', 'category']).size().reset_index(name='count'),
#    x='month_year',
#    y='count',
#    color='category',
#    title='Post Activity Over Time',
#    labels={'count': '# of Posts', 'month_year': 'Date', 'category': 'Category'},
#    height=400,
#    color_discrete_map=category_colors
#)
#fig_posts_over_time.update_xaxes(dtick="M1", tickformat="%b\n%Y")
#st.plotly_chart(fig_posts_over_time, use_container_width=True)

# Wordcloud
keywords_filtered = keywords[keywords['authorFullName'] == selected_author] if selected_author != 'All' else keywords[keywords['cnt'] > 3]
word_freq = keywords_filtered.set_index('KEYWORD')['cnt'].to_dict()
colormap = mcolors.ListedColormap(['#0069c8', '#85c9fe', '#ff2a2b', '#feaaab'])

# Title
title_text = "Keyword Frequency"
if selected_author != 'All':
    title_text += f" for {selected_author}"


st.markdown(f"<br>**{title_text}**", unsafe_allow_html=True)

wordcloud = WordCloud(width=2000, height = 500, background_color ='black', colormap=colormap).generate_from_frequencies(word_freq)
wordcloud_array = wordcloud.to_array()

plt.figure(figsize=(10, 5), frameon=False)
plt.imshow(wordcloud_array, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

gemini_html = f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{base64.b64encode(open(keboola_gemini, "rb").read()).decode()}" style="width: 40px; margin-top: 20px; margin-bottom: 10px;"></div>'

st.markdown(f"{gemini_html}", unsafe_allow_html=True)

data_gemini = data_filtered.loc[:, ['category', 'text', 'likesCount', 'commentsCount']]

_, col, _ = st.columns(3)
if selected_author != 'All':
    if col.button("Analyze the content strategy with Gemini", use_container_width=True):
        json_data = data_gemini.to_json(orient='records')
        generated_text = generate(json_data)
        st.write(generated_text)
else:
    st.info("Select a specific author to analyze the content strategy with Gemini.")