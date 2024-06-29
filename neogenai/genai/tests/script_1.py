from dotenv import load_dotenv
import os
from genai.model.newsbot_services import NewsBot
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()
embedding_model = OpenAIEmbedding(model='text-embedding-ada-002')
newsbot = NewsBot("NEWSBOT_NEUTRALIZE", os.environ['OPENAI_MODEL'], embedding_model=embedding_model)

article = """
Title: "Israel rescues 4 hostages in operation that kills over 270 Gazans"
Text: "Cheers erupted on a beach in Israel as the lifeguard announced the news that Israeli special forces rescued the hostages from Gaza on Saturday. It was the biggest hostage rescue raid of the Israel-Hamas war and also one of the bloodiest Israeli operations of the war. NPR's producer Anas Baba was nearby and ran for cover with other Palestinians.\nGaza's Health Ministry says at least 274 people were killed and 698 wounded. NPR's Daniel Estrin joins us from Tel Aviv, and a warning that this reporting will include graphic descriptions of violence. Good morning.\nIt took place at 11:30 in the morning on Saturday. It was a raid by Israeli special operations forces on two residential apartment buildings in Central Gaza. One of the buildings held a 26-year-old Israeli woman, and the other building held three Israeli male hostages. They were all held in locked rooms in apartment buildings. And there was a gun battle with their Palestinian guards when the Israeli officers came to rescue them. One of the Israeli officers was killed in that gun battle. The officers then left the building with the hostages that came under heavy Palestinian fire, and they left Gaza by helicopter from the Mediterranean shore. This was an operation that they had practiced for weeks using models of the apartment buildings, and they carried out this raid in broad daylight for that element of surprise. All these details were given to reporters by the army spokesman, who acknowledged that there were high Palestinian casualties in this raid. He blamed Hamas for holding hostages among civilians.\nThey were all captured on October 7 when Hamas launched this war. They were captured from the Nova music festival. One of the most well known of them was 26-year-old Noa Argamani. She was filmed as she was being taken on a motorcycle by Hamas into Gaza. She was calling out for help. Her mother, who was born in China, is hospitalized in Israel with advanced cancer, and she had been pleading to see her daughter while she is still alive. Two other security guards who were working at the festival were captured that day and rescued yesterday. One of them was a recent immigrant from Russia, and his parents flew to Israel to see him when he was rescued. So you just see some of the diverse backgrounds of the families touched by this. One 22-year-old Israeli man who was rescued - his father actually died hours before he was rescued, and so he didn't live long enough to hear the news that his own son was freed."
"""
article = """
Title: Who are the released hostages? Text: Who are the released Israeli hostages?\nIsraeli Defense Forces (IDF) rescued Noa Argamani, 26, Almog Meir Jan, 22, Andrei Kozlov, 27, and Shlomi Ziv, 41,in a daylight raid in central Gaza on 8 June. The IDF said they were freed during a "high-risk, complex mission" from two separate buildings in the Nuseirat area. It happened as scores of Palestinians were killed by Israeli attacks in the same area.\nFernando Marman, 60, and Louis Har, 70, were rescued during fighting in the city of Rafah, in southern Gaza. They had been kidnapped from Kibbutz Nir Yitzhak along with three other members of their family, all of whom were released in November 2023. They are Mia Leimberg, 17, her mother Gabriela, 59, and Mia's aunt Clara Marman, 63.\nHamas also released another Russian-Israeli on the same grounds. Roni Krivoi, 25, was working as a sound engineer at the Supernova music festival when he was kidnapped."""
generated_text = newsbot.create_completion(user_input=article)
print()
