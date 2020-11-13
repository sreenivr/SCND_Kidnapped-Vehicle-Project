/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  std::default_random_engine gen;
  
  // Create the particles and set the initial positions based on 
  // the given x, y, theta and their uncertainties, append to vector.
  for (int i = 0; i < num_particles; ++i)
  {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    
    particles.push_back(p);
  }
  
  is_initialized = true; // Set initialized to true
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  std::default_random_engine gen;

  // Predict where each particle will be after delta_t given 
  // the velocity and yaw rate of the car
  double pred_delta_x;      // Change in particles 'x' based on motion model
  double pred_delta_y;      // Change in particles 'y' based on motion model
  double pred_delta_theta;  // Change in particles 'theta' based on motion model
  for (int i = 0; i < num_particles; ++i)
  {
    // If yaw rate is close to zero
    if(fabs(yaw_rate) < 0.0001)
    {
      pred_delta_x = (velocity * delta_t * cos(particles[i].theta));
      pred_delta_y = (velocity * delta_t * sin(particles[i].theta));
    }
    else
    {
      pred_delta_x = (velocity / yaw_rate)*(sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      pred_delta_y = (velocity / yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      pred_delta_theta = (yaw_rate * delta_t);
    }
    // Update the position of the particle based on the motion model
    // also add some gussian noise to the position.
    particles[i].x += pred_delta_x + dist_x(gen);
    particles[i].y += pred_delta_y + dist_y(gen);
    particles[i].theta += pred_delta_theta + dist_theta(gen);
  }
  
  return;
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  double minDist; // distance to nearest-neighbor
  int minDistId;  // Id of the predicted measurement corresponding to minDist
  double distance = 0; 
  
  for (unsigned int i = 0; i < observations.size(); ++i) // Iterate through each observations
  {
    minDist = std::numeric_limits<double>::max(); // Initialize with max double value
    minDistId = -1; 
    
    for (unsigned int j = 0; j < predicted.size(); ++j) // Iterate through each predicted landmarks
    {
      distance = dist(observations[i].x, observations[i].y,
                      predicted[j].x, predicted[j].y); // Compute the distance 
      // If distance less than the nearest-neighbor so far, update minDist and corresponding minDistId
      if(distance < minDist) 
      {
        minDist = distance;
        minDistId = j;
      }
    }
    //std::cout << minDist << " " << minDistId << std::endl;
    observations[i].id = predicted[minDistId].id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (unsigned int p = 0; p < particles.size(); ++p) // Iterate through each particle and update its weight
  {
    std::vector<LandmarkObs> observableLandmarks;     // Landmarks that are within the sensor_range. 
    std::vector<LandmarkObs> transformedObservations; // Observations that are transformed to map coordinates
    
    // Use only the landmarks that are within the sensor_range from the particle
    for (unsigned int lm = 0; lm < map_landmarks.landmark_list.size(); ++lm)
    {
      // Compute the distance from the particle to each landmark in the map
      double distance = dist(particles[p].x, particles[p].y, 
                             map_landmarks.landmark_list[lm].x_f, map_landmarks.landmark_list[lm].y_f);
      // If the distance to the landmark from the particle is within the sensor range,
      // add this landmark to our observableLandmarks collection.
      if(distance < sensor_range)
      {
        LandmarkObs lmo;
        lmo.id = map_landmarks.landmark_list[lm].id_i;
        lmo.x = map_landmarks.landmark_list[lm].x_f; 
        lmo.y = map_landmarks.landmark_list[lm].y_f;
        
        observableLandmarks.push_back(lmo);
      }
    }    
    
    // Transform Observations from Car's coordinates to map coordinates
    for (unsigned int i = 0; i < observations.size(); ++i)
    {
        double x, y;
        x = observations[i].x * cos(particles[p].theta) - observations[i].y * sin(particles[p].theta) + particles[p].x;
        y = observations[i].x * sin(particles[p].theta) + observations[i].y * cos(particles[p].theta) + particles[p].y;

      	LandmarkObs tmp;
        tmp.id = observations[i].id;
        tmp.x = x;
        tmp.y = y;
        transformedObservations.push_back(tmp); // Set ID=0 for now, dataAssociation() will set the correct ID
    }
    
    // Associate the observations with landmarks on the map using nearest-neighbor
    dataAssociation(observableLandmarks, transformedObservations);
    
    double weight;
    weight = 1;
    // Compute and update the weight the particle using Multivariate Gaussian distribution
    for (unsigned int j = 0; j < transformedObservations.size(); ++j)
    {
        unsigned int k = 0;

        // Note: If we had stored the index of the observableLandmarks in 
        // dataAssociation() instead of the id, we could avoid the below loop
        // to search the corresponding landmark.
        
        // find the landmark corresponding to observation
        for (k = 0; k < observableLandmarks.size(); ++k)
        {
          if(observableLandmarks[k].id == transformedObservations[j].id)
          {       
            break;
          }
        }
        
        weight *= multiv_prob(std_landmark[0], std_landmark[1], 
                              transformedObservations[j].x, transformedObservations[j].y,
                              observableLandmarks[k].x, observableLandmarks[k].y
                              );
    }
    particles[p].weight = weight;
  } // End of for loop iterating over particles
}


void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // Check std::default_random_engine vs std::mt19937
  std::random_device rd{};
  std::mt19937 gen{rd()}; 
 
  weights.clear(); // clear the weights vector
  
  // Create a vector of particle weights that can be passed to std::discrete_distribution
  for(int i = 0; i < num_particles; ++i)
  {
    weights.push_back(particles[i].weight);
  }
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  
  std::vector<Particle> resampledParticles;
  for(int i = 0; i < num_particles; ++i)
  {
    int sample = dist(gen);
    resampledParticles.push_back(particles[sample]);
  }
  particles.clear();
  particles = resampledParticles;
}


void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

