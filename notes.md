# Notes on NBA Salaries

## Cleaning

career field goals is missing from too many players (~35%) so setting it to 0.

9/25/2022 went from 4500 data points to 2500 data points after cleaning. Added 500 rows by getting rid of efg column.

## Visualization

If you only include players who went to college, the player with the greatest career revenue is Shaquille O'Neal, at 300 million, followed by Tim Duncan at 240 million, and Carmelo Anthony at 232 million.

If you include people who never went to college, you're looking at Kevin Garnet with 343 million and Kobe Bryant with 328 million. Then you have Shaq, followed by Dirk Nowitsky at 240 million. Since we only have data until 2018, we miss out on Lebron, who has only earned 237 million at this point.

## Regression

It does not really make sense to predict the season salary from career stats, since career stats are the same but season stats are different. It would make more sense to predict the total career revenue, since that corresponds to the total career stats that we have.
