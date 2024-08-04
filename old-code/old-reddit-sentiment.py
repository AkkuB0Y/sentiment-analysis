from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# get this from ACTUAL twitter later!
post = "@akkub0y wow such a fantastic day :)"
post = "&lt;!-- SC_OFF --&gt;&lt;div class=\"md\"&gt;&lt;p&gt;Hey guys!&lt;/p&gt;\n\n&lt;p&gt;We are using this megathread for students to track which companies are cancelling jobs amid COVID-19 panic.&lt;/p&gt;\n\n&lt;p&gt;Good luck to all members of this community searching for a job next term.&lt;/p&gt;\n\n&lt;p&gt;Thank Mr. Goose&lt;/p&gt;\n\n&lt;p&gt;&amp;#x200B;&lt;/p&gt;\n\n&lt;p&gt;EDIT: This for Summer 2020 too, sorry for not clarifying. &lt;/p&gt;\n&lt;/div&gt;&lt;!-- SC_ON --&gt;"

# preprocessing the post, removing standard substrings!
post = post.replace("&lt;!-- SC_OFF --&gt;&lt;div class=\"md\"&gt;&lt;p&gt;", "")
post = post.replace("&lt;/p&gt;\n\n&lt;p&gt;", "")
post = post.replace("&lt;/p&gt;\n&lt;/div&gt;&lt;!-- SC_ON --&gt;;", "")

print(post)

# the actual model from huggingface!
roberta = "cardiffnlp/twitter-roberta-base-sentiment-latest"

model = AutoModelForSequenceClassification.from_pretrained(roberta)

tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

# sentiment analysis
encoded_post = tokenizer(post, return_tensors='pt')

output = model(**encoded_post)

scores = output[0][0].detach().numpy()
scores = softmax(scores)
print(scores)

for i in range(len(scores)):

    l = labels[i]
    s = scores[i]
    print(l,s)