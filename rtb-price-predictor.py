import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
import base64
import time

plt.style.use('ggplot')

###################################################################################################################
# Page Layout settings
###################################################################################################################

# Set Page Layout
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {1200}px;
        padding-top: {5}rem;
        padding-right: {1}rem;
        padding-left: {1}rem;
        padding-bottom: {10}rem;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


# Set title
st.markdown("# Real Time Ad Bidding Price Predictor")
st.markdown("A web app that predicts a reserve CPM price for Publishers.")
st.markdown("""
Programmatic advertising uses your predicted value CPM rates to select the highest bidder. When your value CPMs are outdated, your networks could be competing with incorrect rates, 
preventing you from maximizing ad revenue. With this predictor, you could increase your revenue by updating your value CPM rates at least once per month. 
You can run optimization tests to determine what the sweet spot for your CPM rates are to maximize ad utilization and revenue.
""")
st.markdown("""
\n The real-time ad bidding price predictor will take in the following inputs to determine what the reserve price should be set at: 
""")
st.code('base_price_predictor = (advertiser_id, order_id, monetization_channel_id, line_item_type_id, site_id, ad_unit_id, geo_id, device_category_id, os_id)')

###################################################################################################################
# Read in file
###################################################################################################################
@st.cache
def get_data():
    AWS_BUCKET_URL =  'https://et3-datasets.s3.amazonaws.com/rtb-price-predictor-app/rtb-ad-dataset-processed.csv'
    df = pd.read_csv(AWS_BUCKET_URL)
    return df.set_index("date")

df = get_data()

###################################################################################################################
#  Sidebar
###################################################################################################################
### Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
col1 = st.sidebar
#col2, col3 = st.beta_columns((2,1))

# Sidebar + Main panel
col1.header('Input Options')

## Sidebar - Number of coins to display
number_input = col1.number_input('Sample', 5)

# Sider inputs
col1.header('Features for Model')
x1 = st.sidebar.multiselect('Advertiser ID', [8,16,79,84,88,90,96,97,139,2089,2634,2635,2636,2637,2638,2639,2640,2641,2642,2643,2644,2645,2646])
x2 = st.sidebar.multiselect('Order ID', [45,  140,  146,  147,  148,  151,  152,  158,  162,  170,  172, 177, 2750, 2751, 2752])
x3 = st.sidebar.multiselect('Monetization Channel ID', [1,2,4,19,21])
x4 = st.sidebar.multiselect('Line Item Type ID', [3,4,8,9,11,19,20])
x5 = st.sidebar.multiselect('Site ID', [342, 343, 344, 345, 346, 347, 348, 349, 350, 351])
x6 = st.sidebar.multiselect('Ad Unit ID', [5050, 5051, 5052, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5060,
													5061, 5062, 5063, 5064, 5065, 5066, 5067, 5068, 5069, 5070, 5071,
													5072, 5073, 5074, 5076, 5078, 5079, 5080, 5081, 5082, 5083, 5084,
													5085, 5086, 5087, 5088, 5089, 5090, 5091, 5092, 5093, 5094, 5095,
													5096, 5097, 5098, 5099, 5100, 5101, 5102, 5103, 5104, 5105, 5106,
													5107, 5108, 5109, 5110, 5111, 5112, 5113, 5114, 5115, 5116, 5117,
													5118, 5119, 5120, 5121, 5122, 5123, 5124, 5125, 5126, 5127, 5128,
													5129, 5130, 5131, 5132, 5133, 5134, 5135, 5136, 5137, 5138, 5139,
													5140, 5141, 5142, 5143, 5144, 5145, 5146, 5147, 5148, 5150, 5151,
													5152, 5153, 5154, 5155, 5156, 5157, 5158, 5159, 5160, 5161, 5162,
													5163, 5164, 5165, 5166, 5167, 5168, 5169, 5170, 5171, 5172, 5173,
													5174, 5175, 5176, 5177, 5178, 5179, 5180, 5181, 5183, 5442, 5443])
x7 = st.sidebar.multiselect('Ad Type ID', [10, 17])
x8 = st.sidebar.multiselect('Geographic ID', [  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13, 14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
												27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39, 41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
        										54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66, 67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
												80,  81,  82,  83,  84,  85,  87,  88,  89,  90,  91,  92,  93, 94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106,
       											107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
												133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
       											159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
       											186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 208, 209, 210, 212, 217,
												220, 221, 222, 223, 224, 225, 226, 258, 305, 306, 308])
x9 = st.sidebar.multiselect('Device Category ID', [1,2,3,4,5])
x10 = st.sidebar.multiselect('Operating System ID', [15,55,56,57,58,59,60])

st.sidebar.button('Run')

col1.header('Contribute')
st.sidebar.info('This is an open source project and you can contribute to the project by adding comments, questions, issues, or pull requests to the source code.')

col1.header('About')
st.sidebar.info('This app is maintained by Eric Tran. You can learn more about me at www.ericttran.com.')

###################################################################################################################
# Body
###################################################################################################################
expander_bar = st.beta_expander("About the Dataset")
expander_bar.markdown("""
* **Data Source:** This data was uploaded to [Kaggle](https://www.kaggle.com/saurav9786/real-time-advertisers-auction) on 2020-06-02 by Saurav Anand. 
* **Data Dictionary:**
\n\n

| Key | Value Type | Definition |
| :- | :- | :- |
| date | string | The date the ad was served on |
| advertiser_id | integer |  each id denotes a different bidder in the auction |
| order_id | integer | each id denotes an order generated at the time of creation. Each order ID is useful to ensure you and others are referring to the same Advertiser campaign. |
| line_item_type_id | integer | line items contain information about how specific ads are served on a website along with pricing and other delivery details. For example: Sponsorship, Standard, Network, Bulk, Price Priority. Line item types also have different priorities, or how a line item competes with other line items. Not all line items have the same importance. Some line items may be contractually guaranteeded to serve or may promise more revenue than others. These can be priotiezed higher than other line items. |
| monetization_channel_id | integer |  it describes the mode through which demand partner integrates with a particular publisher - it can be header bidding (running via prebid.js), dynamic allocation, exchange bidding, direct etc |
| site_id | integer | each id denotes a different website |
| ad_type_id | integer | each id denotes a different ad_type. These can be display ads , video ads, text ads etc |
| ad_unit_id | integer | each id denotes a different ad unit (one page can have more than one ad units) | 
| geo_id | integer |  each id denotes a different country. our maximum traffic is from english speaking countries |
| device_category_id | integer | each id denoted a different device_category like desktop , mobile, tablet |
| os_id | integer | each id denotes a different operating system for mobile device category only (android , ios etc) . for all other device categories, osid will correspond to not_mobile |
| integration_type_id | integer |  it describes how the demand partner is setup within a publisher's ecosystem - can be adserver (running through the publisher adserver) or hardcoded |
| total_impressions | float | measurement column measuring the impressions for the particular set of dimensions |
| viewable_impressions | float | Number of impressions on the site that were viewable out of all measurable impressions. A display ad is counted as viewable if at least 50% of its area was displayed on screen for at least one second |
| measureable_impressions | float | Impressions that were measurable by Active View out of the total number of eligible impressions. This value should generally be close to 100%. For example, an impression that is rendering in a cross-domain iframe may not be measurable. Active View is a technology on YouTube and certain Display Network websites and mobile apps that allows Google Ads to determine if your ad is viewable by potential customers. |
| total_revenue | float | measurement column measuring the revenue for the particular set of dimensions |
| revenue_share_percentage | float | not every advertiser gives all the revenue to the publisher. They charge a certain share for the services they provide. This captures the fraction of revenue that will actually reach the publishers pocket |
| cpm | float | CPM: cost-per-thousand impressions. A measure that calculates the Adverrtiser's cost for 1000 impressions

""")

sample = df.sample(number_input)

st.dataframe(df.describe().T)

if st.checkbox('Show Raw Data'):
	st.dataframe(sample)
	st.write('Data Dimensions: ' + str(sample.shape[0]) + ' rows and ' + str(sample.shape[1]) + ' columns.')

st.markdown("---")

###################################################################################################################
# Data Visualizations
###################################################################################################################
st.subheader('CPM Price Distribution by Feature')
columns = ['advertiser_id', 'site_id', 'line_item_type_id', 'monetization_channel_id', 'ad_type_id', 'device_category_id', 'os_id', 'day_of_week']

def get_visualizations():
	fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 15))
	for idx, feat in  enumerate(columns):
		sns.boxplot(x=feat, y='CPM', data=df, ax=axes[idx // 2, idx % 2],  showfliers=False,)
		axes[idx // 2, idx % 2].set_xlabel(feat)
		axes[idx // 2, idx % 2].set_ylabel('CPM')

st.pyplot(get_visualizations())
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("---")

###################################################################################################################
# Modeling
################################################################################################################### 
import pickle
import boto3
s3client = boto3.client('s3')
response = s3client.get_object(Bucket='et3-datasets', Key='rtb-price-predictor-app/rf_reg.pkl')
body = response['Body'].read()
model = pickle.loads(body)

## Modeling
st.subheader('CPM Price Prediction')
st.write('Select values for the feature inputs from the sidebar to run the model.')

if all([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]):
	with st.spinner(text='In progress...'):
		time.sleep(5)
	model_prediction = model.predict(np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]).T)
	st.success("The CPM per 1000 impressions is: ${:.3f}".format(model_prediction[0]))

	st.text("")
	st.markdown("**Feature Values**")
	st.text('- Advertiser ID: {}'.format(x1[0]))
	st.text('- Order ID: {}'.format(x2[0]))
	st.text('- Monetization Channel ID: {}'.format(x3[0]))
	st.text('- Line Item Type ID: {}'.format(x4[0]))
	st.text('- Site ID: {}'.format(x5[0]))
	st.text('- Ad Unit ID: {}'.format(x6[0]))
	st.text('- Ad Type ID: {}'.format(x7[0]))
	st.text('- Geographic ID: {}'.format(x8[0]))
	st.text('- Device Category ID: {}'.format(x9[0]))
	st.text('- Operating System ID: {}'.format(x10[0]))


	# Model Info
	expander_bar = st.beta_expander("About the Model")
	expander_bar.markdown("""**Model:** Random Forest Regressor""")
	expander_bar.markdown("""**Accuracy:** 73%""")
	expander_bar.markdown("""**RMSE:** $0.545""")	
	expander_bar.markdown("""
	Tree-based algorithm. Random forests are an ensemble method where hundreds (or thousands) of individual decision trees are fit to boostrap re-samples of the original dataset, with each tree being allowed to use a random selection of N variables, where N is the major configurable parameter of this algorithm.

	Ensembling many re-sampled decision trees serves to reduce their variance, producing more stable estimators that generalize well out-of-sample. Random forests are extrememly hard to over-fit, are very accurate, generalize well, and require little tuning, all of which are desirable properties in a predictive algorithm.
	""")
	expander_bar.image('./assets/random-forest-tree.png')
	expander_bar.markdown("""Feature Importance""")
	expander_bar.image('./assets/rf_reg-feature-importance.png')
	expander_bar.markdown("""Cumulative Feature Importance""")
	expander_bar.image('./assets/rf_reg-feature-importance-cumulative.png')
	expander_bar.markdown("""Single Tree of Random Forest""")
	expander_bar.image('./assets/rf_reg-decision-tree.png')

	st.balloons()


###################################################################################################################
## Closing Notes
################################################################################################################### 
