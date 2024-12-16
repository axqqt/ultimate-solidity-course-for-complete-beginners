// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

// 1️⃣ Create a Twitter Contract 
// 2️⃣ Create a mapping between user and tweet 
// 3️⃣ Add function to create a tweet and save it in mapping
// 4️⃣ Create a function to get Tweet 
// 5️⃣ Add array of tweets 

contract Twitter {

    struct Tweet {
        string title;
        string content; // Renamed for better understanding
    }

    mapping(address => Tweet) private tweets; // Correct mapping syntax
    Tweet[] public allTweets; // Array to store all tweets

    // Function to create and save a tweet
    function createTweet(string memory _title, string memory _content) public {
        Tweet memory newTweet = Tweet(_title, _content);
        tweets[msg.sender] = newTweet; // Save to mapping
        allTweets.push(newTweet); // Save to array
    }

    // Function to get the tweet of a specific user
    function getTweet(address user) public view returns (string memory, string memory) {
        Tweet memory userTweet = tweets[user];
        return (userTweet.title, userTweet.content);
    }

    // Function to get all tweets (in array format)
    function getAllTweets() public view returns (Tweet[] memory) {
        return allTweets;
    }
}
