const natural = require('natural');
const { TfIdf } = natural;
const fs = require('fs');

// Load audiobooks data from JSON file
const audiobooks = JSON.parse(fs.readFileSync('audiobooks.json', 'utf8'));

// Convert audiobook data into a textual representation
function getFeatureText(audiobook) {
  return `${audiobook.description} ${audiobook.genre} ${audiobook.author} ${audiobook.tags.join(" ")}`;
}

// Convert text into numerical vectors using TF-IDF
const tfidf = new TfIdf();
audiobooks.forEach(book => tfidf.addDocument(getFeatureText(book)));

function getTFIDFVector(index) {
  const vector = {};
  tfidf.listTerms(index).forEach(term => vector[term.term] = term.tfidf);
  return vector;
}

// Compute cosine similarity
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0, magA = 0, magB = 0;

  Object.keys(vecA).forEach(term => {
    if (vecB[term]) dotProduct += vecA[term] * vecB[term];
    magA += Math.pow(vecA[term], 2);
  });

  Object.keys(vecB).forEach(term => {
    magB += Math.pow(vecB[term], 2);
  });

  return dotProduct / (Math.sqrt(magA) * Math.sqrt(magB));
}

// Recommend books based on user history
function recommendBooks(userHistory, k = 3) {
  let historyVectors = userHistory.map(id => {
    let index = audiobooks.findIndex(book => book.id === id);
    return getTFIDFVector(index);
  });

  // Compute the average vector from user's history
  let avgVector = {};
  historyVectors.forEach(vec => {
    Object.keys(vec).forEach(term => {
      avgVector[term] = (avgVector[term] || 0) + vec[term] / historyVectors.length;
    });
  });

  // Compute similarity with all books
  let similarities = audiobooks.map((book, i) => ({
    book,
    similarity: cosineSimilarity(avgVector, getTFIDFVector(i))
  }));

  // Sort by highest similarity & filter out already played books
  return similarities
    .filter(item => !userHistory.includes(item.book.id)) // Exclude already played books
    .sort((a, b) => b.similarity - a.similarity) // Sort by similarity
    .slice(0, k) // Pick top k books
    .map(item => item.book);
}

// Get recommendations
// User history: IDs of books the user recently listened to
const userHistory = [1];
console.log(recommendBooks(userHistory, 3));
