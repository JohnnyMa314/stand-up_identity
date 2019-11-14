import datetime
import os
import pandas as pd
import re
import requests
import wikipedia
from bs4 import BeautifulSoup


# get wikipedia data for comedians
def get_comedian_summary(comedian):
	print(comedian)
	info = [comedian]

	# reading their wikipedia page
	try:
		wik = wikipedia.WikipediaPage(comedian)
	except:
		# try to add (comedian) to comedian name
		try:
			info = [comedian + ' (comedian)']
			wik = wikipedia.WikipediaPage(comedian + ' (comedian)')
		except:
			for _ in range(4):
				info.append('Not Found')
			return info

	# summary
	summary = wik.summary
	if summary is not None:
		info.append(summary)
	else:
		info.append('Not Found')

	# early life
	early_sections = ['Early life', 'Early Life', 'Early life and family', 'Life and career', 'Early life and background', 'Early life and education']
	for section in early_sections:
		early = wik.section(section)
		if early is not None and early:
			info.append(early.replace('\n', ' ').replace("\'", ""))
			break
	if early is None or not early:
		info.append('Not Found')

	# career
	career = wik.section('Career')
	if career is not None and career:
		info.append(career.replace('\n', ' ').replace("\'", ""))
	else:
		info.append('Not Found')

	# personal life
	life = wik.section('Personal life')
	if life is not None and life:
		info.append(life.replace('\n', ' ').replace("\'", ""))
	else:
		info.append('Not Found')

	return info


def get_age(wiki_info):
	print(wiki_info[0])
	summary = wiki_info[1]

	split = re.search('(\d{4})', summary)
	if split is not None:
		birth_year = split[0]
		current_year = datetime.datetime.now().year
		age = int(current_year) - int(birth_year)
	else:
		age = "N/A"

	return age


def get_gender(wiki_info):
	print(wiki_info[0])
	# combine all none-personal life info
	dump = wiki_info[1] + wiki_info[2] + wiki_info[3]

	# count male, female, and other pronouns using regex.
	male = '\\bhe\\b|\\bhis\\b|\\bhim\\b'
	male_count = len(re.findall(male, dump, re.IGNORECASE))
	female = '\\bshe\\b|\\bher\\\b|\\bhers\\b'
	female_count = len(re.findall(female, dump, re.IGNORECASE))
	other = '\\bthey\\b|\\btheir\\b|\\bthemselves\\b'
	other_count = len(re.findall(other, dump, re.IGNORECASE))

	# return highest count as gender
	var = {male_count: "male", female_count: "female", other_count: "other"}
	gender = var.get(max(var))

	return gender


def get_ethnicity(wiki_info):
	print(wiki_info[0])
	dump = wiki_info[1] + wiki_info[2] + wiki_info[3]

	# get list of African-Americans from wikipedia
	url = requests.get('https://en.wikipedia.org/wiki/Category:African-American_stand-up_comedians').text
	soup = BeautifulSoup(url, 'html.parser')
	AA = soup.select(".mw-category-group a")
	for i in range(0, len(AA)):
		AA[i] = re.findall(r'title="(.*?)"', str(AA[i]))

	# Asian American Counter
	asian = r'\bChinese\b|\bAsian\b|\bAsian-American\b|\bKorean\b|\bJapanese|\bIndia\b|\bFilipino\b'
	asian_counter = len(re.findall(asian, dump, re.IGNORECASE))

	# Hispanic
	latinx = r'\bLatin\b|\bMexican\b|\bPuerto\b|\bHispanic\b|\bColombian\b|\bSpanish\b|\bDomincan\b'
	latinx_counter = len(re.findall(latinx, dump, re.IGNORECASE))

	# Jewish
	jewish = r'\bJew\b|\bJewish\b|\bIsrael\b'
	jewish_counter = len(re.findall(jewish, dump, re.IGNORECASE))

	# return highest count as gender
	var = {asian_counter: "asian", latinx_counter: "latinx", jewish_counter: 'jewish'}
	race = var.get(max(var))

	# return highest count as gender
	if [wiki_info[0]] in AA:
		return 'African-American'
	elif max(var) > 2:
		return race
	else:
		return "White"


def get_orientation(wiki_info):
	# get list of LGBT from wikipedia
	print(wiki_info[0])
	url = requests.get('https://en.wikipedia.org/w/index.php?title=Category:LGBT_comedians&pageuntil=Post%2C+Sue-Ann%0ASue-Ann+Post#mw-pages').text
	soup = BeautifulSoup(url, 'html.parser')
	LGBT = soup.select(".mw-category-group a")
	for i in range(0, len(LGBT)):
		LGBT[i] = re.findall(r'title="(.*?)"', str(LGBT[i]))

	# return highest count as gender
	if [wiki_info[0]] in LGBT:
		return 'LGBTQ'
	else:
		return 'Straight-Cis'


def main():
	dir = '/Users/johnnyma/Documents/'
	os.chdir(dir)

	net = pd.read_csv('List.csv')

	comedians = net.Comedian
	comedians = list(dict.fromkeys(comedians))

	# get wikipedia info from comedians
	info = []
	for i in range(0, len(comedians)):
		info.append(get_comedian_summary(comedians[i]))

	info_df = pd.DataFrame(info)
	info_df.columns = ['comedian', 'summary', 'early_life', 'career', 'personal_life']
	info_df.to_csv('raw_wiki.csv')

	# filling in data for each identity category
	df = pd.DataFrame(columns=['comedian', 'age', 'gender', 'orientation', 'ethnicity'])
	for i in range(0, len(info)):
		df.loc[i, 'comedian'] = info[i][0]
		df.loc[i, 'age'] = get_age(info[i])
		df.loc[i, 'gender'] = get_gender(info[i])
		df.loc[i, 'orientation'] = get_orientation(info[i])
		df.loc[i, 'ethnicity'] = get_ethnicity(info[i])

	df.to_csv('wiki_identity.csv')


# if __name__ == '__main__':
# main()

wiki_info = get_comedian_summary('Hasan Minhaj')
