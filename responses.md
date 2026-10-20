# PS4 written responses

Answer all three questions. Replace each line `TODO: Write your response.` with
one short paragraph (about 3–5 sentences), keeping the blank line after the
question. Use your results from `outputs/` to support your answers, and give
the numbers you use.

1. **Which digits get mixed up?**
   Open `confusion-table.csv`, the confusion table for logistic regression. Find
   the two pairs of digits the model mixes up most often. Count a pair both ways:
   for 2 and 7, add the 2s read as 7 to the 7s read as 2. Report the two pairs and
   their counts. Then look at the examples in `misread-digits.png` and explain
   what it is about the shapes of these digits that makes them easy to confuse.

   TODO: Write your response.

2. **Logistic regression or the Perceptron?**
   Report the overall test accuracy of each method from the rows of
   `accuracy.csv` with digit `all`. Which method reads more digits correctly?
   Both methods draw a straight-line boundary for each digit, but they train it
   differently. Use what you know about how each one updates its weights, and
   about data that no straight line can separate perfectly, to explain the
   difference.

   TODO: Write your response.

3. **Can the plant trust the logbook?**
   From `logbook-summary.csv`, report the fraction of single test digits read
   correctly and the fraction of four-digit entries read correctly (the entries
   are the test digits taken four at a time, in order). Compare the entry
   accuracy with the digit accuracy raised to the fourth power, which is the
   chance of reading all four digits correctly if each digit's mistakes happen
   independently. Why are whole entries read correctly less often than single
   digits? Look at `logbook-page.png` and explain what one misread digit can do
   to a recorded value, and what this means for an engineer who uses the
   digitized logbook.

   TODO: Write your response.
