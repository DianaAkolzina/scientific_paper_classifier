import pandas as pd
import openai
import string

# Set your API key
openai.api_key = ""

topics  = [ "referendum", "ballot", "campaign", "caucus", "constituency", "debate", "democrat", "deregulation", "diplomacy", "election", "embassy", "extremism", "fascism", "gerrymander", "globalism", "government", "governor", "hegemony", "immigration", "inauguration", "judiciary", "jurisdiction", "legislation", "legislature", "liberalism", "lobbying", "magistrate", "manifesto", "marxism", "nationalism", "nato", "nepotism", "opposition", "ordinance", "parties", "partisanship", "plurality", "populism", "premier", "primary", "privatization", "proclaim", "proletariat", "prorogue", "protest", "reactionary", "recall", "recount", "redistricting", "republican" ]


# Helper function to generate text
def generate_text(topic, type):
    if type == "scientific":
        system_message = "You are a scientific text generator. Argue against the pseudoscientific topics and defend the scientific topics use evience examples and so on "
        user_prompt = f"Generate a detailed scientific text in one paragraph of 400 words connected to the topic of {topic} but some branch of it. Make it unique and detailed, make it sound like the main part of a scientific publication. Don't use general words and avoid making conclusions. sound like a human scientist"
    else:
        system_message = "You are a pseudoscientific author who tries to defend their argument. Make creative claims about events."
        user_prompt = f"Generate a pseudoscientific text or conspiracy theory in a form of 1 paragraph of 400 words on the topic of {topic} or related matter. It should be from the perspective of a pseudoscientist who makes up the facts instead of using actual science. Make it unique and detailed, make it sound like the main part of a pseudoscientific article. Don't use general words and avoid making conclusions. It should be very inventive and detailed, make creative claims about events."

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=400,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()

# Helper function to remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Function to generate texts and add to dataframe
def generate_texts_df(topics, num_samples=700):
    scientific_texts = []
    pseudoscientific_texts = []

    for i in range(num_samples):
        topic = topics[i % len(topics)]

        scientific_text = generate_text(topic, "scientific")
        print(scientific_text)
        scientific_texts.append(remove_punctuation(scientific_text))

        pseudoscientific_text = generate_text(topic, "pseudoscientific")
        print(pseudoscientific_text)
        pseudoscientific_texts.append(remove_punctuation(pseudoscientific_text))

    scientific_df = pd.DataFrame({"Processed_Text": scientific_texts, "Label": 0})
    pseudoscientific_df = pd.DataFrame({"Processed_Text": pseudoscientific_texts, "Label": 1})
    combined_df = pd.concat([scientific_df, pseudoscientific_df], ignore_index=True)
    combined_df = combined_df.dropna(subset=['Label'])

    return combined_df

# Generate texts and create dataframe
combined_df = generate_texts_df(topics, num_samples=200)

# Save the dataframe to a CSV file
combined_df.to_csv("generated_texts_18.csv", index=False)
