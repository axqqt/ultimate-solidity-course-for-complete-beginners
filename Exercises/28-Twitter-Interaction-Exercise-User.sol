// SPDX-License-Identifier: MIT

// 1️⃣ Save UserProfile to the mapping in the setProfile() function
// HINT: don't forget to set the _displayName and _bio

pragma solidity ^0.8.0;

contract Profile {
    struct UserProfile {
        string displayName;
        string bio;
    }

    address tester = 0x90e608a9e92C564733b0a8771be3160199119c81;
    
    mapping(address => UserProfile) public profiles;

    function setProfile(string memory _displayName, string memory _bio) public {
        // CODE HERE 👇
        profiles[tester] = UserProfile(_displayName,_bio);
    }

    function getProfile(address _user) public view returns (UserProfile memory) {
        return profiles[_user];
    }
}