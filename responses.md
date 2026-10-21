# PS4 written responses

Answer all three questions. Each question has two parts, (a) and (b). Replace
the line `TODO: Write your response.` with your answers, labeled (a) and (b).
One to three sentences per part is enough. Give the numbers you use from the
files in `outputs/`. Keep the blank line after each question; the checker uses
it to find your answer.

1. **Which digits get mixed up?**
   Use `confusion-matrix.csv`, the confusion matrix for logistic regression.
   - **(a)** Find the two pairs of digits the model mixes up most often,
     counting each pair both ways: for 2 and 7, add the 2s read as 7 (row
     `true_digit` 2, column `read_as_7`) to the 7s read as 2 (row `true_digit` 7,
     column `read_as_2`). Report the two pairs and their counts.
   - **(b)** Find examples of your two pairs in `misread-digits.png` (each row
     shows one direction, such as 5 read as 3). A person reads most of these
     digits easily. Use the score x̂ᵀθ (each pixel value times its weight,
     added up over all the pixels, plus the bias) to explain why the model still
     mixes them up.

   TODO: Write your response.

2. **Logistic regression or the Perceptron?**
   Both methods draw a straight-line boundary for each digit, but they train it
   differently.
   - **(a)** Report the overall testing accuracy of each method, from the rows of
     `accuracy.csv` with digit `all`. Which method reads more digits correctly?
   - **(b)** For every digit, the Perceptron still makes mistakes on the training
     images after its 20 passes: it has not found a straight line that separates
     that digit from the others. Use how each method updates its parameters to
     explain the difference in accuracy.

   TODO: Write your response.

3. **Can the plant trust the logbook?**
   Use `logbook-summary.csv` and `logbook-page.png`. The four-digit entries in
   the summary are the testing digits taken four at a time, in order.
   - **(a)** Report the fraction of single testing digits read correctly and the
     fraction of four-digit entries read correctly. Compute the digit accuracy
     raised to the fourth power, and use it to explain why whole entries are
     read correctly less often than single digits.
   - **(b)** In `logbook-page.png`, misread entries are in red. List each one with
     its true value and the value read. Name one check an engineer could run on
     the digitized values that would catch the largest of these errors, and say
     whether it would also catch the smallest.

   TODO: Write your response.
