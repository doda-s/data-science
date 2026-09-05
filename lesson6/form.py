"""
Formulario de apresentacao da turma (class_materials/formulario_alunos.xlsx):
42 respostas, 15 perguntas de um Google Forms (texto livre, multipla escolha
e checkbox), lidas pelo pandas como strings/objetos misturados.

Tratamento aplicado para deixar o dataframe pronto para analise:
- Colunas renomeadas (os cabecalhos originais sao as perguntas inteiras, com
  espacos/quebras de linha inconsistentes).
- Texto livre padronizado (strip + espacos internos colapsados).
- Tokenizacao apenas onde faz sentido (respostas tipo checkbox/lista:
  transporte, areas de interesse, jogos, series, bibliotecas). Perguntas
  dissertativas (definicao de IA/ML, expectativa) sao tratadas como texto,
  nao como lista, ja que nao ha itens discretos para tokenizar.
- Respostas de "sim/nao" em texto livre viram booleanas.
- Escalas de autoavaliacao (Python, Git) viram categoricas ordenadas + score.
- As 4 perguntas de quiz (com tipos mistos int/str no Excel) sao padronizadas
  para string e comparadas com o gabarito, gerando uma pontuacao 0-4.
"""

import re

import pandas

pandas.set_option("display.max_columns", None)
pandas.set_option("display.width", 200)


def clean_text(value):
    """Strip + colapsa espacos/quebras de linha internas. NaN vira pandas.NA."""
    if pandas.isna(value):
        return pandas.NA
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if text else pandas.NA


def count_words(text):
    return 0 if pandas.isna(text) else len(text.split())


NEGATIVE_PATTERN = re.compile(r"^(n[ãa]o\b|nunca\b)", re.IGNORECASE)


def is_positive_answer(raw_value):
    """
    True a menos que a resposta comece claramente com uma negacao (nao/nunca).
    Respostas de sim/nao em texto livre nao tem parsing perfeito -- aqui se
    prioriza o padrao mais comum nos dados ("Nao ...") em vez de tentar cobrir
    toda variacao possivel de fraseado.
    """
    text = clean_text(raw_value)
    return False if pandas.isna(text) else NEGATIVE_PATTERN.match(text) is None


YES_PREFIX_PATTERN = re.compile(
    r"^(sim|gosto)\b[\s,.!]*\s*(sim\b[\s,.!]*)?", re.IGNORECASE
)
LIST_SEPARATOR_PATTERN = re.compile(r"\n|;|/|&|\se\s", re.IGNORECASE)


def tokenize_items(raw_value):
    """
    Extrai uma lista de itens de uma resposta livre tipo "Sim, item1, item2 e
    item3": remove o prefixo de confirmacao, normaliza separadores (virgula,
    "e", ";", "/", "&", quebra de linha) para virgula e devolve os tokens em
    minusculo, sem pontuacao nas pontas.

    Nao usa clean_text no texto inteiro antes de separar: varias respostas
    usam quebra de linha como unico separador de item (ex.: "Hades\\nR.E.P.O\\n
    Stardew Valley"), e colapsar espacos/quebras cedo demais juntaria tudo
    num token so. O whitespace so e colapsado dentro de cada token, depois
    de já ter sido usado para dividir a lista.
    """
    if pandas.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    text = YES_PREFIX_PATTERN.sub("", text, count=1)
    text = LIST_SEPARATOR_PATTERN.sub(",", text)
    tokens = []
    for part in text.split(","):
        token = re.sub(r"\s+", " ", part).strip(" .").lower()
        if token and token not in ("sim", "nao", "não"):
            tokens.append(token)
    return tokens


TRANSPORT_KEYWORDS = [
    ("Nenhum", ["nenhum", "não utilizo", "nao utilizo"]),
    ("A pé", ["a pé", "a pe"]),
    ("Bicicleta", ["bicicleta", "bike"]),
    ("Moto", ["moto"]),
    ("Van", ["van"]),
    ("Ônibus", ["ônibus", "onibus"]),
    ("Carona", ["carona"]),
    # "proprio"/"veiculo"/"automovel" sao tratados como carro: no contexto das
    # respostas (deslocamento para a faculdade), sao usados como sinonimo de
    # carro particular, nunca de moto/bicicleta.
    ("Carro", ["carro", "veículo", "veiculo", "automóvel", "automovel", "próprio", "proprio"]),
]


def classify_transport_fragment(fragment):
    normalized = fragment.strip(" .").lower()
    for canonical, keywords in TRANSPORT_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return canonical
    return normalized.title() if normalized else None


def tokenize_transport_modes(raw_value):
    """
    Algumas respostas usam mais de um meio de transporte (ex.: 'Moto, carro',
    'Ônibus pra vir e carona pra voltar.'), entao cada resposta vira uma lista
    de modos padronizados em vez de uma unica categoria.
    """
    text = clean_text(raw_value)
    if pandas.isna(text):
        return []
    modes = []
    for fragment in re.split(r",|\se\s", text, flags=re.IGNORECASE):
        canonical = classify_transport_fragment(fragment)
        if canonical and canonical not in modes:
            modes.append(canonical)
    return modes


def standardize_quiz_answer(value):
    """Unifica respostas mistas (int/str/NaN vindas do Excel) para string."""
    if pandas.isna(value):
        return pandas.NA
    if isinstance(value, (int, float)):
        return str(int(value))
    return clean_text(value)


df = pandas.read_excel("./class_materials/formulario_alunos.xlsx")

print("\nestrutura original (antes da limpeza)\n".upper())
print(f"{df.shape[0]} respostas, {df.shape[1]} perguntas")
print(df.dtypes)

print("\nrenomeando colunas (cabecalhos originais sao as perguntas inteiras)\n".upper())
df.columns = [
    "Transporte_Raw",
    "Areas_Interesse_Raw",
    "Gosta_Jogos_Raw",
    "Gosta_Series_Raw",
    "Experiencia_Profissional_Raw",
    "Definicao_IA",
    "Definicao_ML",
    "Expectativa_Disciplina",
    "Experiencia_Python_Raw",
    "Bibliotecas_Utilizadas_Raw",
    "Familiaridade_Git_Raw",
    "Quiz_Indexacao_Lista_Raw",
    "Quiz_List_Comprehension_Raw",
    "Quiz_Mutabilidade_Lista_Raw",
    "Quiz_Import_Alias_Raw",
]
print(df.columns.tolist())

print("\ntransporte: tokenizado em modos padronizados (multi-resposta)\n".upper())
df["Transporte_Modos"] = df["Transporte_Raw"].apply(tokenize_transport_modes)
print(df["Transporte_Modos"].head(10))

print("\nareas de interesse: tokenizadas em lista\n".upper())
df["Areas_Interesse"] = df["Areas_Interesse_Raw"].apply(tokenize_items)
print(df["Areas_Interesse"].head(10))

print("\ngosta de jogos: booleana + lista de favoritos\n".upper())
df["Gosta_Jogos"] = df["Gosta_Jogos_Raw"].apply(is_positive_answer)
df["Jogos_Favoritos"] = df.apply(
    lambda row: tokenize_items(row["Gosta_Jogos_Raw"]) if row["Gosta_Jogos"] else [],
    axis=1,
)
print(df[["Gosta_Jogos", "Jogos_Favoritos"]].head(10))

print("\ngosta de series: booleana + lista de favoritas\n".upper())
df["Gosta_Series"] = df["Gosta_Series_Raw"].apply(is_positive_answer)
df["Series_Favoritas"] = df.apply(
    lambda row: tokenize_items(row["Gosta_Series_Raw"]) if row["Gosta_Series"] else [],
    axis=1,
)
print(df[["Gosta_Series", "Series_Favoritas"]].head(10))

print("\nexperiencia profissional: booleana + texto limpo\n".upper())
"""
Texto dissertativo (ex.: 'Sim, trabalho como desenvolvedor fullstack na Save
Company...') nao vira lista: nao ha itens discretos para tokenizar, apenas
uma frase. Tokenizar aqui so quebraria a frase sem gerar valor analitico.
"""
df["Tem_Experiencia_Profissional"] = df["Experiencia_Profissional_Raw"].apply(is_positive_answer)
df["Experiencia_Profissional_Detalhes"] = df["Experiencia_Profissional_Raw"].apply(clean_text)
print(df[["Tem_Experiencia_Profissional"]].value_counts())

print("\nrespostas dissertativas: texto limpo + contagem de palavras\n".upper())
for column in ["Definicao_IA", "Definicao_ML", "Expectativa_Disciplina"]:
    df[column] = df[column].apply(clean_text)
    df[f"{column}_Palavras"] = df[column].apply(count_words)
print(df[["Definicao_IA_Palavras", "Definicao_ML_Palavras", "Expectativa_Disciplina_Palavras"]].describe())

print("\nexperiencia com python: categorica ordenada + score\n".upper())
PYTHON_LEVELS = [
    "Nunca tive contato com Python",
    "Já tive contato, mas lembro de pouca coisa",
    "Consigo criar programas simples",
    "Consigo desenvolver programas de dificuldade intermediária",
    "Utilizo Python com frequência e consigo desenvolver aplicações mais complexas",
]
df["Experiencia_Python_Raw"] = df["Experiencia_Python_Raw"].apply(clean_text)
df["Experiencia_Python"] = pandas.Categorical(df["Experiencia_Python_Raw"], categories=PYTHON_LEVELS, ordered=True)
df["Experiencia_Python_Nivel"] = df["Experiencia_Python"].cat.codes
print(df["Experiencia_Python"].value_counts().reindex(PYTHON_LEVELS))

print("\nbibliotecas utilizadas: tokenizadas em lista + flags por biblioteca\n".upper())
df["Bibliotecas_Utilizadas"] = df["Bibliotecas_Utilizadas_Raw"].apply(
    lambda value: [] if pandas.isna(clean_text(value)) else [item.strip() for item in str(value).split(",")]
)
LIBRARY_FLAGS = {
    "Usa_NumPy": "numpy",
    "Usa_Pandas": "pandas",
    "Usa_Scikit_Learn": "scikit-learn",
    "Usa_Matplotlib_Outra": "matplotlib",
    "Nenhuma_Biblioteca": "nenhuma",
}
for flag_column, keyword in LIBRARY_FLAGS.items():
    df[flag_column] = df["Bibliotecas_Utilizadas"].apply(
        lambda items, kw=keyword: any(kw in item.lower() for item in items)
    )
print(df[list(LIBRARY_FLAGS.keys())].sum())

print("\nfamiliaridade com git/github: categorica ordenada + score\n".upper())
GIT_LEVELS = [
    "Já usei pouco e com dificuldade",
    "Uso o básico: clone, add, commit, push",
    "Uso com desenvoltura: branches, merge, resolução de conflito",
]
df["Familiaridade_Git_Raw"] = df["Familiaridade_Git_Raw"].apply(clean_text)
df["Familiaridade_Git"] = pandas.Categorical(df["Familiaridade_Git_Raw"], categories=GIT_LEVELS, ordered=True)
df["Familiaridade_Git_Nivel"] = df["Familiaridade_Git"].cat.codes
print(df["Familiaridade_Git"].value_counts().reindex(GIT_LEVELS))

print("\nquiz de python: padronizando respostas mistas (int/str) e conferindo o gabarito\n".upper())
QUIZ_ANSWER_KEY = {
    "Quiz_Indexacao_Lista": "3",
    "Quiz_List_Comprehension": "[0, 1, 4, 9]",
    "Quiz_Mutabilidade_Lista": "4",
    "Quiz_Import_Alias": "Nenhuma diferença funcional; a segunda apenas define um apelido",
}
for column, correct_answer in QUIZ_ANSWER_KEY.items():
    df[column] = df[f"{column}_Raw"].apply(standardize_quiz_answer)
    df[f"{column}_Correta"] = df[column] == correct_answer

df["Quiz_Pontuacao"] = df[[f"{column}_Correta" for column in QUIZ_ANSWER_KEY]].sum(axis=1)
print(df[[f"{column}_Correta" for column in QUIZ_ANSWER_KEY] + ["Quiz_Pontuacao"]].sum())
print(f"\nMedia de acertos: {df['Quiz_Pontuacao'].mean():.2f} / 4")

print("\ndados ausentes apos a limpeza\n".upper())
print(df.isna().sum())

print("\nestrutura final\n".upper())
print(df.dtypes)

OUTPUT_PATH = "./lesson6/output/formulario_alunos_limpo.csv"
import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nDataframe padronizado salvo em {OUTPUT_PATH}")
