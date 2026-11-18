# PS4 reference responses

Model answers to the three written questions. The numbers come from the
reference outputs; a correct solution gives the same numbers.

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

   (a) The model mixes up 4 and 9 most often: 51 fours were read as 9 and 48
   nines were read as 4, for 99 mix-ups. Next are 3 and 5: 22 threes were read as
   5 and 54 fives were read as 3, for 76 mix-ups. The largest single count, 58
   twos read as 8, is not in either pair: only 10 eights were read as 2, so 2 and
   8 total 68 mix-ups.

   (b) A 4 and a 9 both have a long stroke down the right side, and a 3 and a 5
   have the same curved bottom half. The score adds up each pixel value times its
   weight, so it depends only on which pixels have ink, not on the shape of the
   strokes. A 4 and a 9 put ink on mostly the same pixels, so for either digit
   the model for 4 and the model for 9 give similar scores. The few pixels where
   the two digits differ decide which score is higher. A small change in where
   the writer puts the ink, such as a slanted stroke or a nearly closed top, can
   then change the reading, even though a person reads the digit easily.

2. **Logistic regression or the Perceptron?**
   Both methods draw a straight-line boundary for each digit, but they train it
   differently.
   - **(a)** Report the overall testing accuracy of each method, from the rows of
     `accuracy.csv` with digit `all`. Which method reads more digits correctly?
   - **(b)** For every digit, the Perceptron still makes mistakes on the training
     images after its 20 passes: it has not found a straight line that separates
     that digit from the others. Use how each method updates its parameters to
     explain the difference in accuracy.

   (a) Logistic regression read 0.8974 of the testing digits correctly (8,974 of
   10,000), and the Perceptron read 0.8683 (8,683 of 10,000). Logistic regression
   reads more digits correctly: 291 more of the 10,000.

   (b) The Perceptron changes its parameters only after a mistake, one image at a
   time. Each mistake adds the image's row of `X` to the parameters, or subtracts
   it: a full-size step, no matter how close the score was to zero. Images read
   correctly have no effect. No straight line separates the digit from the
   others, so the mistakes never stop, and after 20 passes the parameters are
   wherever the last few mistakes left them. Each logistic regression step uses
   all the training images at once: images read wrongly or with a score near zero
   move the parameters the most, and images read correctly with a large score
   barely move them. So the logistic regression parameters reflect every training
   image, not just the last few mistakes, and the resulting boundary reads more
   testing digits correctly.

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

   (a) Logistic regression read 0.8974 of the single digits correctly, but only
   0.6672 of the four-digit entries (1,668 of 2,500). An entry is correct only
   when all four of its digits are correct. If each digit is read correctly with
   probability 0.8974, independent of the other three, all four are correct with
   probability 0.8974⁴ = 0.6486. That is much lower than 0.8974 and close to the
   measured 0.6672.

   (b) Three of the 20 entries are misread: the 06:00 reactor temperature, 342.5 K,
   was read as 342.6 K; the 08:00 reactor pressure, 1046 kPa, was read as
   1066 kPa; and the 09:00 reactor temperature, 351.2 K, was read as 253.7 K. To
   catch the largest error, an engineer could flag any reactor temperature that
   differs by more than 10 K from the readings an hour before and an hour after.
   The value 253.7 K is more than 90 K below the 08:00 and 10:00 readings,
   348.6 K and 347.8 K, so the check catches 253.7 K. The check would not catch
   342.6 K, because an error of 0.1 K is much smaller than the normal hourly
   change of about 3 K.
