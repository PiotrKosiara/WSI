from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# ============================================================
# 1. Zdefiniowanie struktury sieci Bayesowskiej (graf DAG)
# ============================================================

# Tworzymy obiekt modelu z trzema zmiennymi losowymi:
# - 'Obecność' wpływa na 'MS_Teams' oraz 'Światło'
# To oznacza, że zmienne 'MS_Teams' i 'Światło' są warunkowo zależne od obecności doktorantki.
model = DiscreteBayesianNetwork([
    ('Obecność', 'Światło'),
    ('Obecność', 'MS_Teams'),
])

# ============================================================
# 2. Definicja rozkładu a priori dla zmiennej 'Obecność'
# ============================================================

# Zmienna 'Obecność' nie ma rodziców, więc definiujemy bezwarunkowy rozkład.
# - 60% czasu doktorantka pracuje zdalnie (nieobecna)
# - 40% czasu jest na uczelni (obecna)
cpd_obecnosc = TabularCPD(
    variable='Obecność',        # nazwa zmiennej
    variable_card=2,            # liczba stanów (nieobecna, obecna)
    values=[[0.6], [0.4]],      # pierwszy wiersz: P(nieobecna), drugi: P(obecna)
    state_names={'Obecność': ['nieobecna', 'obecna']}  # przypisanie etykiet dla czytelności
)

# ============================================================
# 3. Definicja CPD dla zmiennej 'MS_Teams' zależnej od 'Obecność'
# ============================================================

# Warunkowy rozkład P(MS_Teams | Obecność):
# - jeśli obecna: P(zalogowana) = 0.8, P(nie_zalogowana) = 0.2
# - jeśli nieobecna: P(zalogowana) = 0.05, P(nie_zalogowana) = 0.95
cpd_teams = TabularCPD(
    variable='MS_Teams',
    variable_card=2,
    values=[
        [0.95, 0.2],  # wiersz 0: P(nie_zalogowana | nieobecna), P(nie_zalogowana | obecna)
        [0.05, 0.8],  # wiersz 1: P(zalogowana | nieobecna), P(zalogowana | obecna)
    ],
    evidence=['Obecność'],      # zmienna warunkująca
    evidence_card=[2],          # liczba stanów zmiennej 'Obecność'
    state_names={
        'MS_Teams': ['nie_zalogowana', 'zalogowana'],
        'Obecność': ['nieobecna', 'obecna']
    }
)

# ============================================================
# 4. Definicja CPD dla zmiennej 'Światło' zależnej od 'Obecność'
# ============================================================

# Warunkowy rozkład P(Światło | Obecność):
# - jeśli obecna: P(włączone) = 0.5, P(wyłączone) = 0.5
# - jeśli nieobecna: P(włączone) = 0.05, P(wyłączone) = 0.95
cpd_swiatlo = TabularCPD(
    variable='Światło',
    variable_card=2,
    values=[
        [0.95, 0.5],  # P(wyłączone | nieobecna), P(wyłączone | obecna)
        [0.05, 0.5],  # P(włączone  | nieobecna), P(włączone  | obecna)
    ],
    evidence=['Obecność'],
    evidence_card=[2],
    state_names={
        'Światło': ['wyłączone', 'włączone'],
        'Obecność': ['nieobecna', 'obecna']
    }
)

# ============================================================
# 5. Dodanie wszystkich CPD do modelu
# ============================================================
model.add_cpds(cpd_obecnosc, cpd_teams, cpd_swiatlo)

# ============================================================
# 6. Sprawdzenie poprawności modelu (czy wszystkie zmienne mają poprawne CPD)
# ============================================================
assert model.check_model()  # Jeśli coś jest nie tak, to zgłosi wyjątek

# ============================================================
# 7. Utworzenie obiektu do wnioskowania probabilistycznego
# ============================================================
inference = VariableElimination(model)  # Algorytm eliminacji zmiennych

# ============================================================
# 8. Obliczenie warunkowego prawdopodobieństwa:
#    P(Światło = ? | MS_Teams = zalogowana)
# ============================================================
posterior = inference.query(
    variables=['Światło'],                # zmienna, o której chcemy się czegoś dowiedzieć
    evidence={'MS_Teams': 'zalogowana'}   # znana obserwacja (dowód)
)

# Wyświetlenie wyników na konsoli
print(posterior)

# ============================================================
# 9. Wizualizacja struktury sieci Bayesowskiej jako graf DAG
# ============================================================
graph = nx.DiGraph()
graph.add_edges_from(model.edges)
pos = nx.spring_layout(graph)
# Tworzenie wykresu sieci
plt.figure(figsize=(6, 4))
nx.draw(
    graph, pos,
    with_labels=True,
    node_size=2000,
    node_color='lightblue',
    font_size=10,
    arrows=True,
    arrowsize=20
)
plt.title("Struktura sieci Bayesowskiej (DAG)")
plt.tight_layout()
plt.show()
