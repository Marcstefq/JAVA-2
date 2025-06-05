import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import numpy as np
from scipy.spatial.distance import euclidean

# Symbole pour CAC 40
symbol = '^FCHI'  # Symbole Yahoo Finance pour le CAC 40

# Année de référence à comparer
reference_year = 2008
reference_end_date = '2008-08-30'  # Fin avril 2008
comparison_year = 2001  # Année spécifique à comparer

# Années à analyser (de 1999 à 2007)
years_to_analyze = list(range(1999, 2008))

# Télécharger et stocker les données pour chaque année
yearly_data = {}
print(f"Téléchargement des données pour les années {years_to_analyze}...")

for year in years_to_analyze:
    yearly_data[year] = yf.download(symbol, start=f'{year}-01-01', end=f'{year}-12-31')
    yearly_data[year]['Normalized'] = yearly_data[year]['Close'] / yearly_data[year]['Close'].iloc[0] * 100
    yearly_data[year]['DayMonth'] = yearly_data[year].index.strftime('%m-%d')
    print(f"Année {year}: {len(yearly_data[year])} jours de trading téléchargés")

# Télécharger les données pour 2008 jusqu'à avril
yearly_data[reference_year] = yf.download(symbol, start=f'{reference_year}-01-01', end=reference_end_date)
yearly_data[reference_year]['Normalized'] = yearly_data[reference_year]['Close'] / yearly_data[reference_year]['Close'].iloc[0] * 100
yearly_data[reference_year]['DayMonth'] = yearly_data[reference_year].index.strftime('%m-%d')
print(f"Année {reference_year} (jusqu'à avril): {len(yearly_data[reference_year])} jours de trading téléchargés")

# Calculer la similarité entre 2008 et 2001
reference_values = yearly_data[reference_year]['Normalized'].values
comparison_values = yearly_data[comparison_year]['Normalized'].values[:len(reference_values)]
distance_2001 = euclidean(reference_values, comparison_values)
print(f"\nDistance entre {reference_year} et {comparison_year}: {distance_2001:.2f}")

# Trouver l'année la plus similaire à 2008 en termes de trajectoire
similarities = {}
reference_len = len(reference_values)

for year in years_to_analyze:
    # Prendre le même nombre de jours pour une comparaison équitable
    values = yearly_data[year]['Normalized'].values[:reference_len]
    
    # Si l'année a moins de jours que la référence, ignorer
    if len(values) < reference_len:
        continue
        
    # Calculer la distance euclidienne (plus petite = plus similaire)
    distance = euclidean(reference_values, values)
    similarities[year] = distance

# Trier les similarités
sorted_similarities = sorted(similarities.items(), key=lambda x: x[1])
most_similar_year = sorted_similarities[0][0]

print("\nRésultats de similarité:")
print(f"1. Année la plus similaire à {reference_year}: {most_similar_year}")
print(f"   Distance: {similarities[most_similar_year]:.2f}")
print(f"2. {comparison_year} (comparaison spécifique)")
print(f"   Distance: {distance_2001:.2f}")
print("\nTop 3 des années les plus similaires:")
for i, (year, distance) in enumerate(sorted_similarities[:3], 1):
    print(f"   {i}. {year} (distance: {distance:.2f})")

# Créer un graphique pour comparer 2008 avec 2001 et l'année la plus similaire
plt.figure(figsize=(15, 10))

# Tracer 2001 en bleu
plt.plot(range(len(yearly_data[comparison_year])), yearly_data[comparison_year]['Normalized'], 
         color='blue', linewidth=2, label=f'{comparison_year}')

# Tracer l'année la plus similaire en vert
plt.plot(range(len(yearly_data[most_similar_year])), yearly_data[most_similar_year]['Normalized'], 
         color='green', linewidth=2, label=f'Plus similaire ({most_similar_year})')

# Tracer 2008 en rouge
plt.plot(range(len(yearly_data[reference_year])), yearly_data[reference_year]['Normalized'], 
         color='red', linewidth=3, label=f'{reference_year} (jusqu\'à avril)')

plt.title(f'Comparaison du CAC 40: {reference_year} vs {comparison_year} et année la plus similaire')
plt.xlabel('Jours de trading séquentiels')
plt.ylabel('Indice normalisé (base 100)')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Créer un graphique pour comparer l'alignement par jour calendaire
# Trouver tous les jours-mois possibles
all_day_months = set()
for year, data in yearly_data.items():
    all_day_months.update(data['DayMonth'])
all_day_months = sorted(all_day_months)

# Créer un DataFrame pour l'alignement calendaire
calendar_df = pd.DataFrame(index=all_day_months)

# Ajouter les données pour chaque année
for year, data in yearly_data.items():
    year_values = {}
    for day in all_day_months:
        mask = data['DayMonth'] == day
        if any(mask):
            year_values[day] = data.loc[mask, 'Normalized'].values[0]
    calendar_df[str(year)] = pd.Series(year_values)

# Créer un graphique avec alignement calendaire
plt.figure(figsize=(15, 10))

# Tracer 2001 en bleu
plt.plot(calendar_df.index, calendar_df[str(comparison_year)], 
         color='blue', linewidth=2, label=f'{comparison_year}')

# Tracer l'année la plus similaire en vert
plt.plot(calendar_df.index, calendar_df[str(most_similar_year)], 
         color='green', linewidth=2, label=f'Plus similaire ({most_similar_year})')

# Tracer 2008 en rouge
plt.plot(calendar_df.index, calendar_df[str(reference_year)], 
         color='red', linewidth=3, label=f'{reference_year} (jusqu\'à avril)')

plt.title(f'Comparaison du CAC 40 par jour calendaire: {reference_year} vs {comparison_year} et année la plus similaire')
plt.xlabel('Jour (format: Mois-Jour)')
plt.ylabel('Indice normalisé (base 100)')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.xticks(calendar_df.index[::20])  # Afficher une étiquette tous les 20 jours
plt.tight_layout()
plt.show()

# Graphique de comparaison des performances
plt.figure(figsize=(12, 8))

# Tracer 2001 en bleu
plt.plot(range(len(yearly_data[comparison_year])), yearly_data[comparison_year]['Normalized'], 
         color='blue', linewidth=2, label=f'{comparison_year}')

# Tracer l'année la plus similaire en vert
plt.plot(range(len(yearly_data[most_similar_year])), yearly_data[most_similar_year]['Normalized'], 
         color='green', linewidth=2, label=f'Plus similaire ({most_similar_year})')

# Tracer 2008 en rouge
plt.plot(range(len(yearly_data[reference_year])), yearly_data[reference_year]['Normalized'], 
         color='red', linewidth=2, label=f'{reference_year} (jusqu\'à avril)')

plt.title(f'CAC 40: Comparaison détaillée {reference_year} vs {comparison_year} et année la plus similaire')
plt.xlabel('Jours de trading séquentiels')
plt.ylabel('Indice normalisé (base 100)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show() 