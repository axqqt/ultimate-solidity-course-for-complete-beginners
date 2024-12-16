// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Profile {
    struct UserProfile {
        string displayName;
        string bio;
    }

    mapping(address => UserProfile) public profiles; // Mapping to store user profiles
    address[] private profileAddresses; // Array to track addresses with profiles

    // Function to set a user's profile
    function setProfile(string memory _displayName, string memory _bio) public {
        // Save the user's profile
        profiles[msg.sender] = UserProfile(_displayName, _bio);

        // Add the user to the profileAddresses array if they are new
        bool exists = false;
        for (uint256 i = 0; i < profileAddresses.length; i++) {
            if (profileAddresses[i] == msg.sender) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            profileAddresses.push(msg.sender);
        }
    }

    // Function to get a specific user's profile
    function getProfile(address _user) public view returns (UserProfile memory) {
        return profiles[_user];
    }

    // Function to get all profiles
    function getAllProfiles() public view returns (UserProfile[] memory) {
        UserProfile[] memory allProfiles = new UserProfile[](profileAddresses.length);
        for (uint256 i = 0; i < profileAddresses.length; i++) {
            allProfiles[i] = profiles[profileAddresses[i]];
        }
        return allProfiles;
    }

    // Optional: Function to get all profile addresses
    function getAllProfileAddresses() public view returns (address[] memory) {
        return profileAddresses;
    }
}
