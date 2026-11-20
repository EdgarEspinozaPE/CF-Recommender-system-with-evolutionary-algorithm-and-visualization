# Conjuntos de datos

Los datos no se incluyen en el repositorio. Deben descargarse de su fuente y
colocarse en esta carpeta con la estructura siguiente:

```
datasets/
  ml-100k/u.data, u.item              MovieLens 100K
  hetrec-lastfm/user_artists.dat,     Last.fm HetRec2011
                artists.dat
  Ratings.csv, Books.csv              Book-Crossing
```

| Conjunto | Fuente |
|---|---|
| MovieLens 100K | https://grouplens.org/datasets/movielens/100k/ |
| Last.fm HetRec2011 | https://grouplens.org/datasets/hetrec-2011/ |
| Book-Crossing | Ziegler et al. (2005). Se usa la versión con `Ratings.csv` (`User-ID;ISBN;Rating`) y `Books.csv` (`ISBN;Title;...`)|

El cuaderno de demostración descarga MovieLens 100K y Last.fm HetRec2011 si no se encuentran
