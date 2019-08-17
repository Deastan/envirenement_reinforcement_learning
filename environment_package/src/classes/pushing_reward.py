
def compute_reward(observations, done):
    # The sign depend on its function.
        total_reward = 0
        
        # # create and update from last position
        # last_position = []
        # last_position.append(self.last_pose.position.x)
        # last_position.append(self.last_pose.position.y)
        # last_position.append(self.last_pose.position.z)

        # # create and update current position
        # current_position = []
        # current_position.append(self.current_pose.position.x)
        # current_position.append(self.current_pose.position.y)
        # current_position.append(self.current_pose.position.z)

        # # create the distance btw the two last vector
        # distance_before_move = self.distance_between_vectors(last_position, self.target_position)
        # distance_after_move = self.distance_between_vectors(current_position, self.target_position)

        last_object_position = [self.last_observation[10], self.last_observation[11], self.last_observation[12]]
        current_object_position = [self.current_observation[10], self.current_observation[11], self.current_observation[12]]

        # create the distance btw the two last vector
        distance_before_move = self.distance_between_vectors(last_object_position, self.target_position)
        distance_after_move = self.distance_between_vectors(current_object_position, self.target_position)
        
        # Give the reward
        if self.out_workspace:
            total_reward -= 20
        else:
            if done:
                total_reward += 1500
            else:
                print("Distance object to target: ", distance_after_move - distance_before_move)
                if(distance_after_move - distance_before_move < 0): # before change... >
                    # print("right direction")
                    total_reward += 2
                    total_reward += (1*distance_after_move)
                #if object didnt move.. bad reward
                elif (abs(distance_after_move - distance_before_move) < 0.001):
                    total_reward -= 15.0
                    
                else:
                    # print("wrong direction")
                    total_reward -= 2.0 # 1.0
                    total_reward -= (1*distance_after_move*8) # 1.0
                    
                    
        # print("REWARD: ", distance_after_move )
        # Time punishment
        total_reward -= 1.0

        return total_reward