# Solveur EF non linéaire – Cylindre creux sous pression interne
Cours de l'option MAAS proposé par Pr. Héloïse DANDIN

## Description du projet

Ce projet consiste à compléter le squelette d’un **solveur par éléments finis (EF) non linéaire**.

Le matériau implémenté est **isotrope**, de type **Saint-Venant–Kirchhoff**, les détails des calculs figurent dans le chapitre 5 de l'ouvrage :

> Bonet & Wood — *Non-Linear Continuum Mechanics for Finite Element Analysis*

---

## Problème étudié

L’objet d’étude est un **cylindre creux** soumis à une **pression interne**.

Les maillages sont fournis au format `.msh` et sont disponibles dans le dossier : /msh


## Exécution du solveur

Pour exécuter le solveur, lancer la commande suivante :

```bash
python M7__test_cylinder.py
```
