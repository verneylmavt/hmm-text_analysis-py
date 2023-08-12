# Sentiment Analysis System for Informal Texts

In the realm of digital communication, particularly social media, there exists a plethora of informal texts—be it a tweet on Twitter or a status on Weibo. This vast sea of data holds valuable insights on the sentiments and emotions of users towards a multitude of subjects, from products to news and more. Tapping into these sentiments has always been a challenging yet rewarding endeavor. It offers benefits like tailoring product recommendations, predicting societal reactions, and even forecasting financial market dynamics.

However, the nature of such social texts sets them apart from conventional language structures. They are inherently informal, often brevity-laden, and ridden with noise in the form of slang, emojis, and unconventional grammar. These characteristics demand the development of specialized machine learning systems to decode and derive sentiments from such texts effectively.

This project aims to embark on this challenge. Here, we design a sequence labeling system tailored for informal texts. Grounding our approach in the principles of Hidden Markov Model (HMM) that we've studied, we aspire to lay the foundational stone for a more intricate, intelligent sentiment analysis system in the future.

For this project, we delve deep into two unique datasets: RU and ES. These datasets, containing labeled and unlabeled data, serve as the backbone for training our model. The labeling system identifies tokens with sentiments - positive, negative, and neutral - by designating them as "Beginning" or "Inside" of the sentiment entity. The "Outside" label is reserved for tokens that lie outside any sentiment context.

In essence, this endeavor is not merely about creating a sentiment analysis tool. It's about understanding the language of social media, respecting its nuances, and extracting invaluable sentiments that can help businesses and individuals make informed decisions.

## Installation

1. **Clone the Repository**

```bash
git clone https://github.com/verneylmavt/50.007-1D_Project.git
```

2. **Navigate to the Project Directory**

For the ES Dataset:

```bash
cd 50.007-1D_Project/ES
```

For the RU Dataset:

```bash
cd 50.007-1D_Project/RU
```

3. **Open your Jupyter Notebook**

For the ES Dataset:

```bash
jupyter notebook es_hmm_dev.ipynb
```

For the RU Dataset:

```bash
jupyter notebook ru_hmm_dev.ipynb
```

4. **Execute the Notebook**

Once the Jupyter Notebook interface opens in your browser, navigate to the Cell menu and select Run All to execute all the cells.

5. **Test F-Score**

Once you have run all the codes, you can check the F-Scores by executing:

```bash
python evalResult.py dev.out <YOUR_FILENAME_HERE>
```

## Dependencies

**Note**: Ensure you have Jupyter installed in your working environment with all the required dependencies. If it's not installed, you can get it using:

```bash
pip install jupyter
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
```

Or alternatively, you can download all the dependecies using:

```bash
pip install -r requirements.txt
```

## Contributors

This project was made possible thanks to the hard work and dedication of the following team members:

- **[Elvern Neylmav Tanny](https://github.com/verneylmavt)**
- **[Dorishetti Kaushik Varma](https://github.com/varmz120)**
- **[Harini Parthasarathy](https://github.com/reenee1601)**

Special thanks to all contributors for their invaluable insights and dedication.