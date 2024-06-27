#!/usr/bin/env python
# coding: utf-8

# ### 1. SVMで学習済みのモデルと単語リストを読み込む

# In[3]:


import pickle

# 保存したモデルをロードする
filename = "svmclassifier.pkl"
loaded_classifier = pickle.load(open(filename, "rb"))

# 単語リストを読み込みリストに保存
basicFormList = []
bffile = "basicFormList.txt"
for line in open(bffile, "r", encoding="utf_8"):
    basicFormList.append(line.strip())
print(len(basicFormList))


# ### 2. クラス・関数の貼り付け
# - 演習5-2と演習5-3の関数を全て貼り付ける

# In[4]:


from janome.tokenizer import Tokenizer

# 単語のクラス
class Word:
    def __init__(self, token):
        # 表層形
        self.text = token.surface

        # 原型
        self.basicForm = token.base_form

        # 品詞
        self.pos = token.part_of_speech
        
    # 単語の情報を「表層系\t原型\t品詞」で返す
    def wordInfo(self):
        return self.text + "\t" + self.basicForm + "\t" + self.pos

# 引数のtextをJanomeで解析して単語リストを返す関数
def janomeAnalyzer(text):
    # 形態素解析
    t = Tokenizer()
    tokens = t.tokenize(text) 

    # 解析結果を1行ずつ取得してリストに追加
    wordlist = []
    for token in tokens:
        word = Word(token)
        wordlist.append(word)
    return wordlist


# In[5]:


import random

# キーワード照合ルールのリスト（keywordMatchingRuleオブジェクトのリスト）
kRuleList = []

# 応答候補のリスト（ResponseCandidateオブジェクトのリスト）
candidateList = []

# キーワード照合ルールのクラス（キーワードと応答の組み合わせ）
class KeywordMatchingRule:
    def __init__(self, keyword, response):
        self.keyword = keyword
        self.response = response

# 応答候補のクラス（応答候補とスコアの組み合わせ）
class ResponseCandidate:
    def __init__(self, response, score):
        self.response = response
        self.score = score
    def print(self):
        print("候補文 [%s, %.5f]" % (self.response, self.score))

# キーワード照合ルールを初期化する関数
def setupKeywordMatchingRule():
    kRuleList.clear()
    for line in open('kw_matching_rule.txt', 'r', encoding="utf_8"):
        arr = line.split(",")    
        # keywordMatchingRuleオブジェクトを作成してkRuleListに追加
        kRuleList.append(KeywordMatchingRule(arr[0], arr[1].strip()))

# キーワード照合ルールを利用した応答候補を生成する関数
def generateResponseByRule(inputText):
    for rule in kRuleList:
        # ルールのキーワードが入力テキストに含まれていたら
        if(rule.keyword in inputText):
            # キーワードに対応する応答文とスコアでResponseCandidateオブジェクトを作成してcandidateListに追加
            cdd = ResponseCandidate(rule.response, 1.0 + random.random())
            candidateList.append(cdd)

# ユーザ入力文に含まれる名詞を利用した応答候補を生成する関数
def generateResponseByInputTopic(inputWordList):
    # 名詞につなげる語句のリスト
    textList = ["は好きですか？", "って何ですか？"]
    
    for w in inputWordList:
        pos2 = w.pos.split(",")
        # 品詞が名詞だったら
        if pos2[0]=='名詞':
            cdd = ResponseCandidate(w.basicForm + random.choice(textList), 
                                    0.7 + random.random())
            candidateList.append(cdd)
            
# 無難な応答を返す関数
def generateOtherResponse():
    # 無難な応答のリスト
    bunanList = ["なるほど", "それで？"]

    # ランダムにどれかをcandidateListに追加
    cdd = ResponseCandidate(random.choice(bunanList), 0.5 + random.random())
    candidateList.append(cdd)


# In[6]:


from collections import Counter

# 単語情報リストを渡すとカウンターを返す関数
def makeCounter(wordList):
    basicFormList = []
    for word in wordList:
        basicFormList.append(word.basicForm)
    # 単語の原型のカウンターを作成
    counter = Counter(basicFormList)
    return counter

# Counterのリストと単語リストからベクトルのリストを作成する関数
def makeVectorList(counterList, basicFormList):
    vectorList = []
    for counter in counterList:
        vector = []
        for word in basicFormList:
            vector.append(counter[word])
        vectorList.append(vector)
    return vectorList

from sklearn import svm

# ネガポジ判定の結果を返す関数
# 引数 text:入力文, classifier：学習済みモデル, basicFormList：ベクトル化に使用する単語リスト
def negaposiAnalyzer(text, classifier, basicFormList):
    # 形態素解析して頻度のCounterを作成
    counterList = []
    wordlist = janomeAnalyzer(text)
    counter = makeCounter(wordlist)
    
    # 1文のcounterだが，counterListに追加
    counterList.append(counter)

    # Counterリストと単語リストからベクトルのリストを作成
    vectorList = makeVectorList(counterList, basicFormList)

    # ベクトルのリストに対してネガポジ判定
    predict_label = classifier.predict(vectorList)

    # 入力文のベクトル化に使用された単語を出力
    for vector in vectorList:
        wl=[]
        for i, num in enumerate(vector):
            if(num==1):
                wl.append(basicFormList[i])
        print(wl)

    # 予測結果を出力
    print(predict_label)

    # 予測結果によって出力を決定
    if predict_label[0]=="1":
        output = "よかったね"
    else:
        output = "ざんねん"

    return output


# ### 3. ネガポジ判定の結果「よかったね」か「ざんねん」を応答候補に追加する関数generateNegaposiResponseを作成

# In[7]:


def generateNegaposiResponse(inputText):
    # ネガポジ判定を実行
    output = negaposiAnalyzer(inputText, loaded_classifier, 
                              basicFormList)
    
    # 応答候補に追加
    cdd = ResponseCandidate(output, 0.7 + random.random())
    candidateList.append(cdd)  


# ### 4. 3つの対話戦略とネガポジ判定で応答候補リストを作成
# - if文でタスク指向の対話を実施
# - else以降で非タスク指向の対話（3つの対話戦略＋ネガポジ）を実施

# In[10]:


#deep_translator(翻訳モジュール)をインポート
from deep_translator import GoogleTranslator

#　オウム返しを応用した関数

def generateOumugaeshi(inputText):
    output = inputText + "ね？"
    output = output.replace("私", "あなた")
    
    #応答候補に追加
    cdd = ResponseCandidate(output, 1.0 + random.random())
    candidateList.append(cdd)

# 関西弁に変換する関数
def generateKansaiben(inputText):
    if "関西弁にして" in inputText:
        num = random.randint(0,5)
        output = inputText + "ねん"
        output = output.replace("私", "わい")
        output = output.replace("お母さん", "おかん")
        output = output.replace("お父さん", "おとん")
        output = output.replace("ありがとう", "おおきに")
        output = output.replace("どうしたの", "どないしたん")
        output = output.replace("すいません", "すんまへん")
        output = output.replace("だめ", "アカン")
        output = output.replace("本当", "ホンマ")
        output = output.replace("面白い", "おもろい")
        output = output.replace("面白くない", "おもんない")
        output = output.replace("違う", "ちゃう")
        output = output.replace("いる", "いてる")
        output = output.replace("関西弁にして", '')
        output = output.replace("　", '')
        if num == 0:
            output = "なんでやねん！"
    
        #応答候補に追加
        cdd = ResponseCandidate(output, 2.0 + random.random())
        candidateList.append(cdd)
        
# 日本語を他の言語に翻訳する関数
def generateHonyaku(inputText):
    if "英語にして" in inputText:
        output = GoogleTranslator(source = "auto", target = "en").translate(inputText)
        output = output.replace("English", "")
        
        #応答候補に追加
        cdd = ResponseCandidate(output, 2.0 + random.random())
        candidateList.append(cdd)

# じゃんけんで使うグローバル変数
count_kati = 0
count_make = 0
count_hiki = 0
# じゃんけんをする関数
def generateJyanken(inputText):
    global count_kati
    global count_make
    global count_hiki
    if "じゃんけん" in inputText:
        num = random.randint(0,2)
        if count_hiki + count_kati + count_make == 5:
            count_kati = 0
            count_make = 0
            count_hiki = 0
            if count_kati == 5:
                output = "君はじゃんけんマスターだ"
            elif count_kati > 2:
                output = "もう少しでじゃんけんマスターだ"
            elif count_kati >=0:
                output = "じゃんけんのセンスがないね\nまたチャレンジしてね!!"
        elif num == 0:
            bot_te = "グー"
            if "グー" in inputText:
                count_hiki += 1
                output = f"あなたの手　グー\nbotの手　グー\nあいこだよ\n{count_kati}勝{count_make}敗{count_hiki}分"
            elif "チョキ" in inputText:
                count_make += 1
                output = f"あなたの手　チョキ\nbotの手　グー\n僕の勝ちだよ\n{count_kati}勝{count_make}敗{count_hiki}分"
            elif "パー" in inputText:
                count_kati += 1
                output = f"あなたの手　パー\nbotの手　グー\n僕の負けだよ\n{count_kati}勝{count_make}敗{count_hiki}分"
        elif num == 1:
            bot_te = "チョキ"
            if "グー" in inputText:
                count_kati += 1
                output = f"あなたの手　グー\nbotの手　チョキ\n僕の負けだよ{count_kati}勝{count_make}敗{count_hiki}分"
            elif "チョキ" in inputText:
                count_hiki += 1
                output = f"あなたの手　チョキ\nbotの手　チョキ\nあいこだよ\n{count_kati}勝{count_make}敗{count_hiki}分"
            elif "パー" in inputText:
                count_make += 1
                output = f"あなたの手　パー\nbotの手　チョキ\n僕の勝ちだよ\n{count_kati}勝{count_make}敗{count_hiki}分"
        elif num == 2:
            bot_te = "パー"
            if "グー" in inputText:
                count_make += 1
                output = f"あなたの手　グー\nbotの手　パー\n僕の勝ちだよ\n{count_kati}勝{count_make}敗{count_hiki}分"
            elif "チョキ" in inputText:
                count_kati += 1
                output = f"あなたの手　チョキ\nbotの手　パー\n僕の負けだよ{count_kati}勝{count_make}敗{count_hiki}分"
            elif "パー" in inputText:
                count_hiki += 1
                output = f"あなたの手　パー\nbotの手　パー\nあいこだよ\n{count_kati}勝{count_make}敗{count_hiki}分"
        
        #応答候補に追加
        cdd = ResponseCandidate(output, 2.0 + random.random())
        candidateList.append(cdd)

# 日別の音楽ランキング（上位5位）を表示する
def genereteOngaku(inputText):
    if "音楽" in inputText and "ランキング" in inputText:
        import requests
        from bs4 import BeautifulSoup
        count = 0
        if "日間" in inputText:
            url = "https://utaten.com/ranking/"
        
            texts = ""
        
            # URLにリクエストしてレスポンスを取得
            res = requests.get(url)
            res.encoding = res.apparent_encoding
        
            # 取得したレスポンスからBeautifulSoupオブジェクトを作る
            soup = BeautifulSoup(res.text, "html.parser")
            texts +=("日間ランキング　参照　UtaTen\n")
            for section in soup.find_all(["h3"]):
                count += 1
                texts += f"{count}位 {section.text}"
                texts += "\n"
                #output = (f"{section.text}")
                if count == 5:
                    break
        elif "週間" in inputText:
            
            url = "https://utaten.com/ranking/?type=weekly"
        
            texts = ""
        
            # URLにリクエストしてレスポンスを取得
            res = requests.get(url)
            res.encoding = res.apparent_encoding
        
            # 取得したレスポンスからBeautifulSoupオブジェクトを作る
            soup = BeautifulSoup(res.text, "html.parser")
            texts +=("週間ランキング　参照　UtaTen\n")
            for section in soup.find_all(["h3"]):
                count += 1
                texts += f"{count}位 {section.text}"
                texts += "\n"
                #output = (f"{section.text}")
                if count == 5:
                    break
        
        elif "月間" in inputText:
            url = "https://utaten.com/ranking/?type=monthly"
        
            texts = ""
        
            # URLにリクエストしてレスポンスを取得
            res = requests.get(url)
            res.encoding = res.apparent_encoding
        
            # 取得したレスポンスからBeautifulSoupオブジェクトを作る
            soup = BeautifulSoup(res.text, "html.parser")
            texts +=("月間ランキング　参照　UtaTen\n")
            for section in soup.find_all(["h3"]):
                count += 1
                texts += f"{count}位 {section.text}"
                texts += "\n"
                #output = (f"{section.text}")
                if count == 5:
                    break
        
        else:
            texts = "日間or週間or月間をつけてください"
        
        cdd = ResponseCandidate(texts, 2.0 + random.random())
        candidateList.append(cdd)
        
def generateMenu(inputText):
    if "menu" in inputText:
        output = """「関西弁にして」を入力で関西弁になります。\n「英語にして」を入力で英語になります。\n「じゃんけん　自分の手」を入力でじゃんけんを行うことができます\n　　　　これは、5回勝負となっています。\n「音楽ランキング」を入力で音楽ランキングが表示されます。\n　　　　これは、日間,週間,月間が選択可能です"""
        cdd = ResponseCandidate(output, 2.0 + random.random())
        candidateList.append(cdd)
        
# 応答文を生成する関数
def generateResponse(inputText):
    
    # 出力文
    output = ""
    
    # 決まったキーワードを含むとき（タスク指向）のときの出力
    #入力文が「時間を教えて」を含んでいたら時間を出力に設定
    if "時間を教えて" in inputText:
        import datetime
        dt_now = datetime.datetime.now()
        dt_now_str = dt_now.strftime("%Y/%m/%d %H:%M")
        output = dt_now_str
    
    elif "夕飯" in inputText:
        num = random.randint(0,2)
        if num == 0:
            output = "寿司"
        elif num == 1:
            output = "天ぷら"
        elif num == 2:
            output = "ステーキ"
    
    #それ以外は非タスク指向（雑談対話）で返す
    else:
   
        # 応答文候補を空にしておく
        candidateList.clear()

        # 形態素解析した後，3つの戦略を順番に実行
        wordlist = janomeAnalyzer(inputText)
        generateResponseByRule(inputText)
        generateResponseByInputTopic(wordlist)
        generateOtherResponse()
        
        # オウム返しの関数を実行
        generateOumugaeshi(inputText)
        generateKansaiben(inputText)
        generateHonyaku(inputText)
        generateJyanken(inputText)
        genereteOngaku(inputText)
        generateMenu(inputText)

        # ネガポジ判定の結果を応答候補に追加
        generateNegaposiResponse(inputText)

        ret="デフォルト"
        maxScore=-1.0

        # scoreが最も高い応答文候補を戻す
        for cdd in candidateList:
            cdd.print()
            if cdd.score > maxScore:
                ret=cdd.response
                maxScore = cdd.score
        output = ret
        
    return output


# ### 5. 出力の確認（ipynbファイル上で）

# In[11]:


#Botへの入力
text = "今日はいい天気ですね"
#text = "menu"
#text = "音楽ランキングを表示して　日間"

setupKeywordMatchingRule()

# システムの出力を生成
##### 自分で入力しよう #####
output = generateResponse(text)

print(output)


# In[ ]:





# In[ ]:

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import settings

app = App(token=settings.SLACK_BOT_TOKEN)

# 入力データからuseridとtextを取得
def body_parser(body):
    # useridの取得
    userid = body["event"]["user"]

    # textの取得
    blocks = body["event"]["blocks"][0]
    elements = blocks["elements"][0]
    texts = elements["elements"][1]
    text = texts["text"]
    text = text.strip()

    return userid, text

# メンションされたときに返答する
@app.event("app_mention")
def event_mention(body, say, logger):
    # 入力からユーザIDとテキストを取得
    userid, text = body_parser(body)

    #システムの出力を生成（オウム返し）
    output = text

    # Slackで返答
    say(f"<@{userid}> {output}")
    
@app.event("message")
def handle_message_events(body, logger):
    logger.info(body)
    
if __name__ == "__main__":
    handler = SocketModeHandler(app, settings.SLACK_APP_TOKEN)
    handler.start()


