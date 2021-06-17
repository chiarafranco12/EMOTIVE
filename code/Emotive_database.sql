-- phpMyAdmin SQL Dump
-- version 5.0.1
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: Jun 07, 2021 at 05:57 PM
-- Server version: 10.4.11-MariaDB
-- PHP Version: 7.4.3

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `Emotive`
--

-- --------------------------------------------------------

--
-- Table structure for table `Administrator`
--

CREATE TABLE `Administrator` (
  `id` int(5) NOT NULL,
  `guser_id` int(5) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `Administrator`
--

INSERT INTO `Administrator` (`id`, `guser_id`) VALUES
(1, 1),
(2, 2);

-- --------------------------------------------------------

--
-- Table structure for table `Emotion`
--

CREATE TABLE `Emotion` (
  `id` int(5) NOT NULL,
  `description_e` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `Emotion`
--

INSERT INTO `Emotion` (`id`, `description_e`) VALUES
(1, 'Angry'),
(2, 'Disgust'),
(3, 'Fear'),
(4, 'Happy'),
(5, 'Neutral'),
(6, 'Sad'),
(7, 'Surprise');

-- --------------------------------------------------------

--
-- Table structure for table `Face`
--

CREATE TABLE `Face` (
  `id` int(5) NOT NULL,
  `path_f` varchar(50) NOT NULL,
  `name_f` varchar(100) DEFAULT NULL,
  `surname_f` varchar(100) DEFAULT NULL,
  `fk_emotion_id` int(5) DEFAULT NULL,
  `fk_guser_id` int(5) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `Face`
--

INSERT INTO `Face` (`id`, `path_f`, `name_f`, `surname_f`, `fk_emotion_id`, `fk_guser_id`) VALUES
(8, 'fear.jpg', 'fear', 'fear', 3, 1),
(9, 'happyboy.jpg', 'happy', 'boy', 4, 1),
(11, 'chiara.jpeg', 'Chiara', 'Franco', 4, 1),
(13, 'sad.jpg', 'sad', 'sad', 6, 3);

-- --------------------------------------------------------

--
-- Table structure for table `GUser`
--

CREATE TABLE `GUser` (
  `id` int(5) NOT NULL,
  `name_u` varchar(50) NOT NULL,
  `surname_u` varchar(50) NOT NULL,
  `username_u` varchar(100) NOT NULL,
  `password_u` varchar(100) NOT NULL,
  `role_id` int(5) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `GUser`
--

INSERT INTO `GUser` (`id`, `name_u`, `surname_u`, `username_u`, `password_u`, `role_id`) VALUES
(1, 'chiara', 'franco', 'chiaretta02', 'sasso123', 1),
(2, 'domenico', 'policastro', 'poli182', 'sasso123', 1),
(3, 'mario', 'rossi', 'm.rossi', 'password', 2);

-- --------------------------------------------------------

--
-- Table structure for table `NUser`
--

CREATE TABLE `NUser` (
  `id` int(5) NOT NULL,
  `birthdate_u` date NOT NULL,
  `email_u` varchar(50) NOT NULL,
  `access_key` varchar(100) NOT NULL,
  `confirmed` tinyint(1) NOT NULL,
  `reason_id` int(5) DEFAULT NULL,
  `guser_id` int(5) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `NUser`
--

INSERT INTO `NUser` (`id`, `birthdate_u`, `email_u`, `access_key`, `confirmed`, `reason_id`, `guser_id`) VALUES
(1, '2000-01-13', 'mario.rossi@gmail.com', 'abc123', 1, 1, 3);

-- --------------------------------------------------------

--
-- Table structure for table `Reason`
--

CREATE TABLE `Reason` (
  `id` int(5) NOT NULL,
  `description_m` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `Reason`
--

INSERT INTO `Reason` (`id`, `description_m`) VALUES
(1, 'school'),
(2, 'work'),
(3, 'personal purposes');

-- --------------------------------------------------------

--
-- Table structure for table `Role_u`
--

CREATE TABLE `Role_u` (
  `id` int(5) NOT NULL,
  `description_r` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `Role_u`
--

INSERT INTO `Role_u` (`id`, `description_r`) VALUES
(1, 'admin'),
(2, 'user');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `Administrator`
--
ALTER TABLE `Administrator`
  ADD PRIMARY KEY (`id`),
  ADD KEY `guser_id` (`guser_id`);

--
-- Indexes for table `Emotion`
--
ALTER TABLE `Emotion`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `Face`
--
ALTER TABLE `Face`
  ADD PRIMARY KEY (`id`),
  ADD KEY `fk_guser_id` (`fk_guser_id`),
  ADD KEY `fk_emotion_id` (`fk_emotion_id`);

--
-- Indexes for table `GUser`
--
ALTER TABLE `GUser`
  ADD PRIMARY KEY (`id`),
  ADD KEY `role_id` (`role_id`);

--
-- Indexes for table `NUser`
--
ALTER TABLE `NUser`
  ADD PRIMARY KEY (`id`),
  ADD KEY `reason_id` (`reason_id`),
  ADD KEY `guser_id` (`guser_id`);

--
-- Indexes for table `Reason`
--
ALTER TABLE `Reason`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `Role_u`
--
ALTER TABLE `Role_u`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `Administrator`
--
ALTER TABLE `Administrator`
  MODIFY `id` int(5) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `Emotion`
--
ALTER TABLE `Emotion`
  MODIFY `id` int(5) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- AUTO_INCREMENT for table `Face`
--
ALTER TABLE `Face`
  MODIFY `id` int(5) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=14;

--
-- AUTO_INCREMENT for table `GUser`
--
ALTER TABLE `GUser`
  MODIFY `id` int(5) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `NUser`
--
ALTER TABLE `NUser`
  MODIFY `id` int(5) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `Reason`
--
ALTER TABLE `Reason`
  MODIFY `id` int(5) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `Role_u`
--
ALTER TABLE `Role_u`
  MODIFY `id` int(5) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `Administrator`
--
ALTER TABLE `Administrator`
  ADD CONSTRAINT `administrator_ibfk_1` FOREIGN KEY (`guser_id`) REFERENCES `GUser` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `Face`
--
ALTER TABLE `Face`
  ADD CONSTRAINT `face_ibfk_1` FOREIGN KEY (`fk_guser_id`) REFERENCES `GUser` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `face_ibfk_2` FOREIGN KEY (`fk_emotion_id`) REFERENCES `Emotion` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `GUser`
--
ALTER TABLE `GUser`
  ADD CONSTRAINT `guser_ibfk_1` FOREIGN KEY (`role_id`) REFERENCES `Role_u` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- Constraints for table `NUser`
--
ALTER TABLE `NUser`
  ADD CONSTRAINT `nuser_ibfk_1` FOREIGN KEY (`reason_id`) REFERENCES `Reason` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `nuser_ibfk_2` FOREIGN KEY (`guser_id`) REFERENCES `GUser` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
