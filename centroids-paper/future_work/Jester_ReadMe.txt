Format:

24983 users in total
Ratings are real values ranging from -10.00 to +10.00 (the value "99" corresponds to "null" = "not rated").
One row per user
The first column gives the number of jokes rated by that user. The next 100 columns give the ratings for jokes 01 - 100.
The sub-matrix including only columns {5, 7, 8, 13, 15, 16, 17, 18, 19, 20} is dense. Almost all users have rated those jokes

Preprocessing (to do):
1) convert data to "user, item, rating" format (csv); you can encode userID from 1 to 24983
2) remove the entries rating=99
3) use min-max to normalize ratings to 1 to 5