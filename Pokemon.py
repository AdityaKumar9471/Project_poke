import streamlit as st
from PIL import Image
from Work import classify
import base64
import tensorflow as tf

# Title and header
st.markdown(
    """
    <div style="text-align: center; color: white; padding: 10px;">
        <h1 style="font-family: 'Comic Sans MS', cursive, sans-serif; color: yellow; text-shadow: 2px 2px #3b4cca;">
            <span style="color: yellow; text-shadow: 2px 2px #3b4cca;">PokéMon</span> Image Classifier
        </h1>
        <h2 style="font-family: 'Verdana', sans-serif; color: yellow; text-shadow: 2px 2px #3b4cca;">
            Snap, Upload, and Identify Your Favorite Pokémon!
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="background-color: white; padding: 20px; border-radius: 10px;">
        <p style="font-size: 18px; color: black;">
            This application allows you to upload an image of a Pokémon, and it will predict which Pokémon it is.
            Simply upload an image using the file uploader below, and our model will handle the rest.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


file=st.file_uploader('', type=['jpg', 'jpeg', 'png'])

model=tf.keras.models.load_model("myMLmodel3.keras")

class_names = ['Abra',
               'Aerodactyl',
               'Alakazam',
               'Alolan Sandslash',
               'Arbok',
               'Arcanine',
               'Articuno',
               'Beedrill',
               'Bellsprout',
               'Blastoise',
               'Bulbasaur',
               'Butterfree',
               'Caterpie',
               'Chansey',
               'Charizard',
               'Charmander',
               'Charmeleon',
               'Clefable',
               'Clefairy',
               'Cloyster',
               'Cubone',
               'Dewgong',
               'Diglett',
               'Ditto',
               'Dodrio',
               'Doduo',
               'Dragonair',
               'Dragonite',
               'Dratini',
               'Drowzee',
               'Dugtrio',
               'Eevee',
               'Ekans',
               'Electabuzz',
               'Electrode',
               'Exeggcute',
               'Exeggutor',
               'Farfetchd',
               'Fearow',
               'Flareon',
               'Gastly',
               'Gengar',
               'Geodude',
               'Gloom',
               'Golbat',
               'Goldeen',
               'Golduck',
               'Golem',
               'Graveler',
               'Grimer',
               'Growlithe',
               'Gyarados',
               'Haunter',
               'Hitmonchan',
               'Hitmonlee',
               'Horsea',
               'Hypno',
               'Ivysaur',
               'Jigglypuff',
               'Jolteon',
               'Jynx',
               'Kabuto',
               'Kabutops',
               'Kadabra',
               'Kakuna',
               'Kangaskhan',
               'Kingler',
               'Koffing',
               'Krabby',
               'Lapras',
               'Lickitung',
               'Machamp',
               'Machoke',
               'Machop',
               'Magikarp',
               'Magmar',
               'Magnemite',
               'Magneton',
               'Mankey',
               'Marowak',
               'Meowth',
               'Metapod',
               'Mew',
               'Mewtwo',
               'Moltres',
               'MrMime',
               'Muk',
               'Nidoking',
               'Nidoqueen',
               'Nidorina',
               'Nidorino',
               'Ninetales',
               'Oddish',
               'Omanyte',
               'Omastar',
               'Onix',
               'Paras',
               'Parasect',
               'Persian',
               'Pidgeot',
               'Pidgeotto',
               'Pidgey',
               'Pikachu',
               'Pinsir',
               'Poliwag',
               'Poliwhirl',
               'Poliwrath',
               'Ponyta',
               'Porygon',
               'Primeape',
               'Psyduck',
               'Raichu',
               'Rapidash',
               'Raticate',
               'Rattata',
               'Rhydon',
               'Rhyhorn',
               'Sandshrew',
               'Sandslash',
               'Scyther',
               'Seadra',
               'Seaking',
               'Seel',
               'Shellder',
               'Slowbro',
               'Slowpoke',
               'Snorlax',
               'Spearow',
               'Squirtle',
               'Starmie',
               'Staryu',
               'Tangela',
               'Tauros',
               'Tentacool',
               'Tentacruel',
               'Vaporeon',
               'Venomoth',
               'Venonat',
               'Venusaur',
               'Victreebel',
               'Vileplume',
               'Voltorb',
               'Vulpix',
               'Wartortle',
               'Weedle',
               'Weepinbell',
               'Weezing',
               'Wigglytuff',
               'Zapdos',
               'Zubat']




if file is not None:
    image=Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)

    classn, confidence_score, description = classify(image, model, class_names, pokemon_descriptions)
    st.write("Image Classified!")
    st.write("## {}".format(classn))
    st.write("### Confidence_Score(%): {}".format(confidence_score))
    st.write("### Description: {}".format(description))

st.markdown(
    """
    ### Model Description


    This is a deep learning model based on computer vision to classify uploaded images into 150 distinct categories. 
    Initially, images are resized to meet the model's input specifications. The model employs transfer learning, utilizing 
    the ResNet101v2 architecture to perform image categorization. At the final stage, a Softmax layer computes the probabilities 
    of the image being one of the 150 Pokémon classes. 

    Feel free to experiment with various Pokémon images!

    """,
    unsafe_allow_html=False
)

st.markdown(
    """
    Developed by Aditya
    """,
    unsafe_allow_html=True
)

st.image("pokemon-party.jpg", use_column_width=True)  # Replace with your image file

